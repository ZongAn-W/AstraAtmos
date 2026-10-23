import React from 'react';
import { useT } from '../i18n';
import { useAnimatedNumber } from './ui/motion';
import { getTrainingStatusMeta } from '../pages/ModelTrainingPage/trainingStatusMeta';

const TrainingProgressMonitor = ({
  progress = 0,
  currentEpoch = 0,
  totalEpochs = 0,
  loss = null,
  eta = '--:--',
  status = 'running',
}) => {
  const t = useT();
  const rawPercent = Math.min(100, Math.max(0, progress));
  // 进度来自轮询，直接赋值会跳格；这里缓动到目标值，视觉上连续推进
  const animatedPercent = useAnimatedNumber(rawPercent, { duration: 640 });
  const percent = Math.min(100, Math.max(0, animatedPercent));
  const statusMeta = getTrainingStatusMeta(status, t);
  const metrics = [
    { label: t('modelTraining.statsEpoch'), value: `${currentEpoch}`, suffix: totalEpochs ? `/ ${totalEpochs}` : '' },
    { label: t('modelTraining.statsLoss'), value: loss !== null && Number.isFinite(loss) ? loss.toFixed(4) : '--' },
    { label: t('modelTraining.statsETA'), value: eta || '--:--' },
  ];

  return (
    <div className="experiment-progress">
      <div className="experiment-progress-head">
        <div>
          <div className="experiment-progress-label">{t('modelTraining.statsProgress')}</div>
          <div className="experiment-progress-value">{percent.toFixed(1)}<span>%</span></div>
        </div>
        <span className="experiment-progress-state" data-status={status}>
          <span className="experiment-progress-state-dot" aria-hidden="true" />
          {statusMeta.label}
        </span>
      </div>
      <div
        className="experiment-progress-track"
        role="progressbar"
        aria-label={t('modelTraining.statsProgress')}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Number(rawPercent.toFixed(1))}
      >
        <span style={{ width: `${percent}%`, transition: 'width 120ms linear' }} />
      </div>
      <div className="experiment-progress-metrics">
        {metrics.map((metric) => (
          <div className="experiment-progress-metric" key={metric.label}>
            <div className="experiment-progress-metric-label">{metric.label}</div>
            <div className="experiment-progress-metric-value">
              <span>{metric.value}</span>
              {metric.suffix ? <small>{metric.suffix}</small> : null}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default TrainingProgressMonitor;
