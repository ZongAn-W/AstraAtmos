/** 共用分析区：选择分析 → 调整条件 → 阅读主图，辅助结果按需展开。 */
import React, { useEffect, useState } from 'react';
import { useOverviewLayout } from './OverviewShell.jsx';
import { pickActiveCard } from './observatoryLayout.js';
import { CARD_STATUS } from './OverviewAdapter.js';
import { reasonText } from './OverviewCard.jsx';
import { analysisDescription } from './analysisGuidance.js';

function GroupTabs({ groups, mode, onModeChange, isZh }) {
  return (
    <div className="analysis-groups" role="group" aria-label={isZh ? '分析方向' : 'Analysis category'}>
      {groups.map((group) => {
        const active = mode === group.id;
        return (
          <button key={group.id} type="button" aria-pressed={active}
            onClick={() => { if (!active) onModeChange?.(group.id); }}>
            {isZh ? group.title.zh : group.title.en}
          </button>
        );
      })}
    </div>
  );
}

function CardSelector({ cards, activeCard, onCardChange, isBoard, isZh }) {
  if (!cards.length) return null;
  return (
    <label className="dock-selector">
      <span>{isZh ? '图表' : 'Chart'}</span>
      <select aria-label={isZh ? '图表' : 'Chart'} value={isBoard ? '' : activeCard || ''}
        onChange={(event) => { if (event.target.value) onCardChange?.(event.target.value); }}>
        {isBoard ? <option value="">{isZh ? '选择单项图表…' : 'Choose an individual chart…'}</option> : null}
        {cards.map((card) => (
          <option key={card.key} value={card.key}>
            {isZh ? card.title.zh : card.title.en}
            {card.state?.status === CARD_STATUS.UNSUPPORTED ? (isZh ? '（不可用）' : ' (unavailable)') : ''}
          </option>
        ))}
      </select>
    </label>
  );
}

export default function AnalysisDock({
  planet = 'earth',
  view = 'observe', groups = [], mode, onModeChange = null,
  cards = [], activeCard = '', onCardChange = null,
  onExpand = null, onCollapse = null, renderCard = null,
  isZh = true, scopeLabel = null, loadingLabel = null,
  conditionsSlot = null, contextNote = null, onRetryCard = null,
  aiSlot = null, pointSlot = null, extraSlot = null, emptyHint = null,
  presentation = 'detail', onPresentationChange = null, boardSlot = null, boardConditionsSlot = null,
}) {
  const { dockHeight, flow } = useOverviewLayout();
  const [showPoint, setShowPoint] = useState(false);
  const [showAi, setShowAi] = useState(false);
  const isBoard = presentation === 'board' && boardSlot !== null;
  const group = groups.find(item => item.id === mode);
  const displayedConditions = isBoard ? boardConditionsSlot : conditionsSlot;
  const selectGroup = (next) => { onPresentationChange?.('board'); onModeChange?.(next); };
  const selectDetail = (next) => { onCardChange?.(next); onPresentationChange?.('detail'); };

  // 显式选择优先；只有选择不属于当前目录时才回退，不能用旧 ref 覆盖用户切图。
  const resolvedCard = pickActiveCard(cards, activeCard);
  useEffect(() => {
    if (resolvedCard && resolvedCard !== activeCard) onCardChange?.(resolvedCard);
  }, [activeCard, resolvedCard, onCardChange]);
  const active = cards.find((card) => card.key === resolvedCard) || null;
  const title = active ? (isZh ? active.title.zh : active.title.en) : '';
  const status = active?.state?.status;
  const showFullChart = view === 'analyze' && status === CARD_STATUS.READY;

  // 筛选栏不再占用图表上方高度，只为标题、图注和辅助入口留出空间。
  // 展开辅助内容只增加正文滚动，不挤压主图。
  const mainChartHeight = flow || !dockHeight
    ? 360
    : Math.max(360, Math.round(dockHeight - 260));

  if (view === 'observe') {
    return (
      <div className="analysis-preview">
        <strong>{title}</strong>
        <button type="button" className="analysis-action" onClick={onExpand}>
          {isZh ? '展开分析 ↗' : 'Open analysis ↗'}
        </button>
      </div>
    );
  }

  return (
    <div className="analysis-dock">
      <aside className="dock-head analysis-controls" aria-label={isZh ? '分析选择与条件' : 'Analysis controls'}>
        <div className="analysis-selection-row">
          <div className="analysis-step">
            <span className="analysis-step__number" aria-hidden="true">1</span>
            <span>{isZh ? '选择分析' : 'Choose an analysis'}</span>
          </div>
          <GroupTabs groups={groups} mode={mode} onModeChange={selectGroup} isZh={isZh} />
          <CardSelector cards={cards} activeCard={resolvedCard} onCardChange={selectDetail} isBoard={isBoard} isZh={isZh} />
        </div>
        {displayedConditions || scopeLabel ? (
          <div className="analysis-conditions-row">
            <div className="analysis-step">
              <span className="analysis-step__number" aria-hidden="true">2</span>
              <span>{isZh ? '分析条件' : 'Analysis settings'}</span>
            </div>
            <div className="analysis-conditions">{displayedConditions}</div>
            {scopeLabel ? <span className="analysis-context">{scopeLabel}</span> : null}
          </div>
        ) : null}
        {contextNote ? <p className="analysis-context-note">{contextNote}</p> : null}
        <button type="button" className="analysis-back" onClick={onCollapse}>
          {isZh ? '返回观测' : 'Back to observing'}
        </button>
      </aside>

      <div className="dock-body">
        <div data-dock-main={isBoard ? `board-${mode}` : active?.key || 'none'} data-dock-chart-height={mainChartHeight}
          aria-busy={!isBoard && (status === CARD_STATUS.LOADING || status === CARD_STATUS.IDLE)}>
          <div className="analysis-page-heading">
            <div className="analysis-chart-heading">
              <h2>{isBoard ? (isZh ? group?.title.zh : group?.title.en) : title || (isZh ? '选择一张图表开始分析' : 'Choose a chart to begin')}</h2>
              <p>{isBoard
                ? (isZh ? '相关图表同屏对照，点击放大可查看细节。' : 'Compare related views together. Expand a chart to inspect it.')
                : analysisDescription(active?.key, isZh, planet)}</p>
            </div>
            {boardSlot !== null ? <div className="analysis-presentation" role="group" aria-label={isZh ? '分析展示方式' : 'Analysis presentation'}>
              <button type="button" aria-pressed={isBoard} onClick={() => onPresentationChange?.('board')}>{isZh ? '组合看板' : 'Theme board'}</button>
              <button type="button" aria-pressed={!isBoard} onClick={() => onPresentationChange?.('detail')}>{isZh ? '单项分析' : 'Individual analysis'}</button>
            </div> : null}
          </div>
          {isBoard ? boardSlot : showFullChart ? renderCard?.(active, { compact: false, height: mainChartHeight }) : (
            <div className="analysis-state" role={status === CARD_STATUS.ERROR ? 'alert' : 'status'}>
              {!active ? (isZh ? '当前分组没有可用图表。' : 'No charts are available in this group.')
                : status === CARD_STATUS.UNSUPPORTED ? reasonText(active.state.reason || active.reason, isZh)
                  : status === CARD_STATUS.ERROR ? (isZh ? '图表加载失败，请重试。' : 'The chart could not be loaded. Please retry.')
                    : (isZh ? '正在加载这张图表…' : 'Loading this chart…')}
              {status === CARD_STATUS.ERROR && onRetryCard ? (
                <button type="button" className="analysis-action" onClick={() => onRetryCard(active.key)}>
                  {isZh ? '重新加载图表' : 'Retry chart'}
                </button>
              ) : null}
              {emptyHint ? <span>{emptyHint}</span> : null}
            </div>
          )}
        </div>

        {!isBoard && (pointSlot || aiSlot) ? (
          <div className="analysis-support">
            {pointSlot ? (
              <details open={showPoint} onToggle={(event) => setShowPoint(event.currentTarget.open)}>
                <summary>{isZh ? '点位与逐日数据' : 'Point and daily data'}</summary>
                {showPoint ? <div className="analysis-support__content" data-dock-slot="point">{pointSlot}</div> : null}
              </details>
            ) : null}
            {aiSlot ? (
              <details open={showAi} onToggle={(event) => setShowAi(event.currentTarget.open)}>
                <summary>{isZh ? 'AI 解读当前图表' : 'AI interpretation'}</summary>
                {showAi ? <div className="analysis-support__content" data-dock-slot="ai">{aiSlot}</div> : null}
              </details>
            ) : null}
          </div>
        ) : null}
        {loadingLabel ? <span className="dock-head__status" role="status">{loadingLabel}</span> : null}
        {extraSlot ? <div data-dock-slot="extra">{extraSlot}</div> : null}
      </div>
    </div>
  );
}
