"""
ORM 数据表模型
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey,
    Index, Integer, LargeBinary, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.engine import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    username: Mapped[str] = mapped_column(String(50), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="user")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # 关系
    uploads: Mapped[list["UploadRecord"]] = relationship(
        "UploadRecord", foreign_keys="UploadRecord.user_id", back_populates="uploader"
    )
    reviewed_uploads: Mapped[list["UploadRecord"]] = relationship(
        "UploadRecord", foreign_keys="UploadRecord.reviewed_by", back_populates="reviewer"
    )
    notifications: Mapped[list["Notification"]] = relationship(
        "Notification", foreign_keys="Notification.user_id", back_populates="user"
    )
    lineage_events: Mapped[list["DatasetLineageEvent"]] = relationship(
        "DatasetLineageEvent", foreign_keys="DatasetLineageEvent.actor_user_id", back_populates="actor"
    )
    user_model_packages: Mapped[list["UserModelPackage"]] = relationship(
        "UserModelPackage", foreign_keys="UserModelPackage.user_id", back_populates="owner"
    )
    training_weight_files: Mapped[list["TrainingWeightFile"]] = relationship(
        "TrainingWeightFile", foreign_keys="TrainingWeightFile.user_id", back_populates="owner"
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email} role={self.role}>"


class UploadRecord(Base):
    __tablename__ = "upload_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    mars_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ls_start: Mapped[float | None] = mapped_column(Float, nullable=True)
    ls_end: Mapped[float | None] = mapped_column(Float, nullable=True)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    file_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    data_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="validating")
    is_public: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    validation_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    reviewed_by: Mapped[int | None] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)

    # 关系
    uploader: Mapped["User"] = relationship(
        "User", foreign_keys=[user_id], back_populates="uploads"
    )
    reviewer: Mapped["User | None"] = relationship(
        "User", foreign_keys=[reviewed_by], back_populates="reviewed_uploads"
    )
    lineage_events: Mapped[list["DatasetLineageEvent"]] = relationship(
        "DatasetLineageEvent",
        foreign_keys="DatasetLineageEvent.upload_id",
        back_populates="upload_record",
        cascade="all, delete-orphan",
    )
    quality_snapshots: Mapped[list["DatasetQualitySnapshot"]] = relationship(
        "DatasetQualitySnapshot",
        foreign_keys="DatasetQualitySnapshot.upload_id",
        back_populates="upload_record",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<UploadRecord id={self.id} file={self.filename} status={self.status}>"


class McdCacheJob(Base):
    __tablename__ = "mcd_cache_jobs"
    __table_args__ = (
        Index("ix_mcd_cache_jobs_upload_type_status", "upload_id", "job_type", "status"),
        Index("ix_mcd_cache_jobs_year_type_status", "mars_year", "job_type", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    upload_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("upload_records.id"), nullable=False, index=True
    )
    mars_year: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    job_type: Mapped[str] = mapped_column(String(40), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending", index=True)
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    cache_version: Mapped[str] = mapped_column(String(40), nullable=False, default="v1")
    artifact_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    upload_record: Mapped["UploadRecord"] = relationship(
        "UploadRecord", foreign_keys=[upload_id]
    )

    def __repr__(self) -> str:
        return f"<McdCacheJob id={self.id} upload={self.upload_id} type={self.job_type} status={self.status}>"


class McdCacheArtifact(Base):
    __tablename__ = "mcd_cache_artifacts"
    __table_args__ = (
        UniqueConstraint(
            "upload_id",
            "cache_type",
            "source_hash",
            "cache_version",
            name="uq_mcd_cache_artifact_version",
        ),
        Index("ix_mcd_cache_artifacts_upload_type_status", "upload_id", "cache_type", "status"),
        Index("ix_mcd_cache_artifacts_year_type_status", "mars_year", "cache_type", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    upload_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("upload_records.id"), nullable=False, index=True
    )
    mars_year: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    cache_type: Mapped[str] = mapped_column(String(40), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="building", index=True)
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    cache_version: Mapped[str] = mapped_column(String(40), nullable=False, default="v1")
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    upload_record: Mapped["UploadRecord"] = relationship(
        "UploadRecord", foreign_keys=[upload_id]
    )

    def __repr__(self) -> str:
        return f"<McdCacheArtifact id={self.id} upload={self.upload_id} type={self.cache_type} status={self.status}>"


class DatasetLineageEvent(Base):
    __tablename__ = "dataset_lineage_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    upload_id: Mapped[int] = mapped_column(Integer, ForeignKey("upload_records.id"), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(40), nullable=False)
    event_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    actor_user_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    actor_role: Mapped[str | None] = mapped_column(String(20), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now, index=True)

    upload_record: Mapped["UploadRecord"] = relationship(
        "UploadRecord", foreign_keys=[upload_id], back_populates="lineage_events"
    )
    actor: Mapped["User | None"] = relationship(
        "User", foreign_keys=[actor_user_id], back_populates="lineage_events"
    )

    def __repr__(self) -> str:
        return f"<DatasetLineageEvent id={self.id} upload_id={self.upload_id} type={self.event_type}>"


class DatasetQualitySnapshot(Base):
    __tablename__ = "dataset_quality_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    upload_id: Mapped[int] = mapped_column(Integer, ForeignKey("upload_records.id"), nullable=False, index=True)
    file_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    file_mtime_ns: Mapped[int] = mapped_column(Integer, nullable=False, default=0, index=True)
    overall_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    grade: Mapped[str] = mapped_column(String(2), nullable=False, default="D")
    missing_rate: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    valid_value_ratio: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    variable_completeness: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    time_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    grid_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    metrics_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    scores_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    issues_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    computed_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now, index=True)

    upload_record: Mapped["UploadRecord"] = relationship(
        "UploadRecord", foreign_keys=[upload_id], back_populates="quality_snapshots"
    )

    def __repr__(self) -> str:
        return f"<DatasetQualitySnapshot id={self.id} upload_id={self.upload_id} overall={self.overall_score}>"


class Notification(Base):
    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    # approved | rejected | revoked | training_oom
    type: Mapped[str] = mapped_column(String(30), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_read: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    related_upload_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    related_training_task_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)

    # 关系
    user: Mapped["User"] = relationship(
        "User", foreign_keys=[user_id], back_populates="notifications"
    )

    def __repr__(self) -> str:
        return f"<Notification id={self.id} user_id={self.user_id} type={self.type} read={self.is_read}>"


class Feedback(Base):
    __tablename__ = "feedbacks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    type: Mapped[str] = mapped_column(String(20), nullable=False)   # "bug" | "suggestion" | "other"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    contact_email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    screenshot_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")  # "pending" | "resolved"
    admin_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    user: Mapped["User | None"] = relationship("User", foreign_keys=[user_id])

    def __repr__(self) -> str:
        return f"<Feedback id={self.id} type={self.type} status={self.status}>"


class UserModelPackage(Base):
    __tablename__ = "user_model_packages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    display_name: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(500), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    param_schema: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    validation_status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    validation_report: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    owner: Mapped["User"] = relationship(
        "User", foreign_keys=[user_id], back_populates="user_model_packages"
    )

    def __repr__(self) -> str:
        return f"<UserModelPackage id={self.id} user_id={self.user_id} name={self.display_name}>"


class TrainingWeightFile(Base):
    __tablename__ = "training_weight_files"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(500), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="ready")
    validation_report: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    owner: Mapped["User"] = relationship(
        "User", foreign_keys=[user_id], back_populates="training_weight_files"
    )

    def __repr__(self) -> str:
        return f"<TrainingWeightFile id={self.id} user_id={self.user_id} file={self.original_filename}>"


class ModelTrainingTask(Base):
    __tablename__ = "model_training_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    model_script: Mapped[str] = mapped_column(String(255), nullable=False)
    model_source: Mapped[str] = mapped_column(String(20), nullable=False, default="official")
    uploaded_model_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("user_model_packages.id"), nullable=True
    )
    uploaded_model_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    hyperparameters: Mapped[str] = mapped_column(Text, nullable=False)  # JSON stored as string
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")  # pending, queued, running, completed, failed, cancelled
    start_time: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
    queued_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    queue_position: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    log_file_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    output_model_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    custom_model_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metrics: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON stored as string
    
    # 实时进度追踪字段
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    current_epoch: Mapped[int] = mapped_column(Integer, default=0)
    total_epochs: Mapped[int] = mapped_column(Integer, default=0)
    current_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    eta: Mapped[str | None] = mapped_column(String(50), nullable=True)
    loss_history: Mapped[str | None] = mapped_column(Text, nullable=True) # JSON: {"train": [], "val": []}

    # Dataset identity. Always generated by the server, never by a client, and
    # deliberately nullable: an unmigrated row must not look like a decision.
    # These are identity columns, not hyperparameters: they must never be merged
    # into ``hyperparameters`` or passed to the training CLI.
    dataset_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    dataset_version: Mapped[str | None] = mapped_column(String(40), nullable=True)
    dataset_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    dataset_identity_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    dataset_snapshot: Mapped[str | None] = mapped_column(Text, nullable=True)

    user: Mapped["User | None"] = relationship("User", foreign_keys=[user_id])
    uploaded_model: Mapped["UserModelPackage | None"] = relationship(
        "UserModelPackage", foreign_keys=[uploaded_model_id]
    )

    def __repr__(self) -> str:
        return f"<ModelTrainingTask id={self.id} script={self.model_script} status={self.status}>"


class TrainingModelTag(Base):
    __tablename__ = "training_model_tags"
    __table_args__ = (
        UniqueConstraint("user_id", "name_key", name="uq_training_model_tag_owner_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    name_key: Mapped[str] = mapped_column(String(192), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)


class TrainingTaskTag(Base):
    __tablename__ = "training_task_tags"

    task_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("model_training_tasks.id", ondelete="CASCADE"), primary_key=True
    )
    tag_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("training_model_tags.id", ondelete="CASCADE"), primary_key=True, index=True
    )


class PredictionAnalysisCache(Base):
    __tablename__ = "prediction_analysis_caches"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "training_task_id",
            "analysis_type",
            "request_hash",
            name="uq_prediction_analysis_cache_scope",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False, index=True
    )
    training_task_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("model_training_tasks.id"),
        nullable=False,
        index=True,
    )
    analysis_type: Mapped[str] = mapped_column(
        String(40), nullable=False
    )
    request_hash: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    artifact_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="computing"
    )
    payload: Mapped[bytes | None] = mapped_column(
        LargeBinary, nullable=True
    )
    lease_token: Mapped[str | None] = mapped_column(
        String(36), nullable=True
    )
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True
    )
    last_error: Mapped[str | None] = mapped_column(
        String(500), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=_now, onupdate=_now
    )

    user: Mapped["User"] = relationship("User", foreign_keys=[user_id])
    training_task: Mapped["ModelTrainingTask"] = relationship(
        "ModelTrainingTask", foreign_keys=[training_task_id]
    )


class PersonalSourceBuildState(Base):
    __tablename__ = "personal_source_build_states"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False, unique=True, index=True)
    signature_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="idle")  # idle | building | ready | failed
    stage: Mapped[str] = mapped_column(String(40), nullable=False, default="idle")  # idle | queued | building_cache | warming_analysis | warming_predict | ready | failed
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    stage_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    built_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)

    user: Mapped["User"] = relationship("User", foreign_keys=[user_id])

    def __repr__(self) -> str:
        return (
            f"<PersonalSourceBuildState user_id={self.user_id} status={self.status} "
            f"signature={self.signature_hash[:8]}>"
        )
