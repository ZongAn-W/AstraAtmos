"""Private, per-user organization of training results; independent of model identity."""

from sqlalchemy import delete, select, text
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.exc import IntegrityError

from database.engine import async_session_maker
from database.models import ModelTrainingTask, TrainingModelTag, TrainingTaskTag

# Two IN clauses or two columns per inserted row remain below SQLite's legacy 999 limit.
QUERY_CHUNK_SIZE = 400


class TagNameConflict(ValueError):
    pass


def normalize_tag_name(name: str) -> str:
    normalized = name.strip()
    if not 1 <= len(normalized) <= 64:
        raise ValueError("Tag name must contain 1 to 64 characters")
    return normalized


def _unique_ids(values):
    values = list(dict.fromkeys(values))
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("Tag and task IDs must be positive integers")
    return values


def _chunks(values):
    for offset in range(0, len(values), QUERY_CHUNK_SIZE):
        yield values[offset:offset + QUERY_CHUNK_SIZE]


async def begin_tag_write(session):
    # SQLite's legacy mode does not start a transaction for SELECT. Acquire the
    # write lock before authorization reads so deletion/ID reuse cannot change
    # tag or task ownership between validation and the final commit.
    await session.execute(text("BEGIN IMMEDIATE"))


async def validate_owned_tags(session, user_id, tag_ids):
    """Validate inside the caller's transaction, also used when creating a task."""
    ids = _unique_ids(tag_ids)
    tags = []
    for chunk in _chunks(ids):
        result = await session.execute(select(TrainingModelTag).where(
            TrainingModelTag.id.in_(chunk), TrainingModelTag.user_id == user_id
        ))
        tags.extend(result.scalars().all())
    if len(tags) != len(ids):
        # Foreign and nonexistent tags deliberately have the same response.
        raise FileNotFoundError("Tag not found")
    return tags


async def add_task_tag_links(session, task_ids, tag_ids):
    rows = []
    for task_id in task_ids:
        for tag_id in tag_ids:
            rows.append({"task_id": task_id, "tag_id": tag_id})
            if len(rows) == QUERY_CHUNK_SIZE:
                await session.execute(insert(TrainingTaskTag).values(rows).on_conflict_do_nothing())
                rows = []
    if rows:
        await session.execute(insert(TrainingTaskTag).values(rows).on_conflict_do_nothing())


class TrainingTagService:
    async def list_tags(self, user_id):
        async with async_session_maker() as session:
            result = await session.execute(select(TrainingModelTag).where(
                TrainingModelTag.user_id == user_id
            ).order_by(TrainingModelTag.name_key, TrainingModelTag.id))
            return list(result.scalars().all())

    async def create_tag(self, user_id, name):
        name = normalize_tag_name(name)
        async with async_session_maker() as session:
            tag = TrainingModelTag(user_id=user_id, name=name, name_key=name.casefold())
            session.add(tag)
            await self._commit_name(session)
            return tag

    async def _owned_tag(self, session, tag_id, user_id):
        result = await session.execute(select(TrainingModelTag).where(
            TrainingModelTag.id == tag_id, TrainingModelTag.user_id == user_id
        ))
        tag = result.scalar_one_or_none()
        if tag is None:
            raise FileNotFoundError("Tag not found")
        return tag

    async def _commit_name(self, session):
        try:
            await session.commit()
        except IntegrityError as exc:
            await session.rollback()
            raise TagNameConflict("A tag with this name already exists") from exc

    async def rename_tag(self, tag_id, user_id, name):
        name = normalize_tag_name(name)
        async with async_session_maker() as session:
            await begin_tag_write(session)
            tag = await self._owned_tag(session, tag_id, user_id)
            tag.name, tag.name_key = name, name.casefold()
            await self._commit_name(session)
            return tag

    async def delete_tag(self, tag_id, user_id):
        async with async_session_maker() as session:
            await begin_tag_write(session)
            tag = await self._owned_tag(session, tag_id, user_id)
            await session.execute(delete(TrainingTaskTag).where(TrainingTaskTag.tag_id == tag.id))
            await session.delete(tag)
            await session.commit()

    async def _accessible_tasks(self, session, task_ids, user_id, is_admin):
        ids = _unique_ids(task_ids)
        tasks = {}
        for chunk in _chunks(ids):
            result = await session.execute(select(ModelTrainingTask).where(ModelTrainingTask.id.in_(chunk)))
            tasks.update((task.id, task) for task in result.scalars())
        if len(tasks) != len(ids):
            raise FileNotFoundError("Task not found")
        if not is_admin and any(task.user_id != user_id for task in tasks.values()):
            raise PermissionError("No permission to access this task")
        return [tasks[task_id] for task_id in ids]

    async def replace_task_tags(self, task_id, tag_ids, user_id, *, is_admin=False):
        async with async_session_maker() as session:
            await begin_tag_write(session)
            tasks = await self._accessible_tasks(session, [task_id], user_id, is_admin)
            tags = await validate_owned_tags(session, user_id, tag_ids)
            own_tag_ids = select(TrainingModelTag.id).where(TrainingModelTag.user_id == user_id)
            await session.execute(delete(TrainingTaskTag).where(
                TrainingTaskTag.task_id == task_id, TrainingTaskTag.tag_id.in_(own_tag_ids)
            ))
            await add_task_tag_links(session, [task_id], [tag.id for tag in tags])
            await session.commit()
            return tasks[0]

    async def update_task_tags(self, task_ids, tag_ids, operation, user_id, *, is_admin=False):
        if operation not in {"add", "remove"}:
            raise ValueError("Operation must be add or remove")
        async with async_session_maker() as session:
            await begin_tag_write(session)
            tasks = await self._accessible_tasks(session, task_ids, user_id, is_admin)
            tags = await validate_owned_tags(session, user_id, tag_ids)
            task_ids, tag_ids = [task.id for task in tasks], [tag.id for tag in tags]
            if operation == "add":
                await add_task_tag_links(session, task_ids, tag_ids)
            else:
                for task_chunk in _chunks(task_ids):
                    for tag_chunk in _chunks(tag_ids):
                        await session.execute(delete(TrainingTaskTag).where(
                            TrainingTaskTag.task_id.in_(task_chunk), TrainingTaskTag.tag_id.in_(tag_chunk)
                        ))
            await session.commit()
            return tasks

    async def get_task_tags(self, task_ids, user_id):
        """Hydrate one caller's labels in bounded queries, without ORM lazy loading."""
        ids = _unique_ids(task_ids)
        tags = {task_id: [] for task_id in ids}
        async with async_session_maker() as session:
            for chunk in _chunks(ids):
                result = await session.execute(select(
                    TrainingTaskTag.task_id, TrainingModelTag.id, TrainingModelTag.name
                ).join(TrainingModelTag, TrainingModelTag.id == TrainingTaskTag.tag_id).where(
                    TrainingTaskTag.task_id.in_(chunk), TrainingModelTag.user_id == user_id
                ).order_by(TrainingModelTag.name_key, TrainingModelTag.id))
                for task_id, tag_id, name in result:
                    tags[task_id].append({"id": tag_id, "name": name})
        return tags
