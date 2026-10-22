import { useState } from 'react';
import { useT } from '../i18n';
import MorphSubmitButton from '../components/ui/MorphSubmitButton';
import JellyToggle from '../components/ui/JellyToggle';
import HoldToDeleteButton from '../components/ui/HoldToDeleteButton';
import './UiKitPage/uikitPage.css';

const DEMO_ROWS = ['demo-row-a', 'demo-row-b'];

function Card({ title, desc, children }) {
  return (
    <section className="uikit-card">
      <h2 className="uikit-card__title">{title}</h2>
      <p className="uikit-card__desc">{desc}</p>
      <div className="uikit-card__stage">{children}</div>
    </section>
  );
}

export default function UiKitPage() {
  const t = useT();
  const [submitCount, setSubmitCount] = useState(0);
  const [jettyOn, setJettyOn] = useState(false);
  const [removed, setRemoved] = useState([]);

  const handleSubmit = async () => {
    await new Promise(resolve => setTimeout(resolve, 700));
    setSubmitCount(count => count + 1);
  };

  return (
    <main className="uikit-page">
      <header className="uikit-page__head">
        <h1 className="uikit-page__title">{t('uikit.title')}</h1>
        <p className="uikit-page__subtitle">{t('uikit.subtitle')}</p>
      </header>

      <div className="uikit-grid">
        <Card title={t('uikit.morphTitle')} desc={t('uikit.morphDesc')}>
          <MorphSubmitButton
            label={t('uikit.morphLabel')}
            loadingLabel={t('uikit.morphLoading')}
            successLabel={t('uikit.morphSuccess')}
            onAction={handleSubmit}
          />
          <p className="uikit-state" data-demo="morph-state">
            {t('uikit.morphState', { n: submitCount })}
          </p>
        </Card>

        <Card title={t('uikit.jellyTitle')} desc={t('uikit.jellyDesc')}>
          <div className="uikit-inline">
            <JellyToggle
              checked={jettyOn}
              onChange={setJettyOn}
              label={t('uikit.jellyLabel')}
            />
            <span className="uikit-state" data-demo="jelly-state" data-checked={jettyOn ? 'true' : 'false'}>
              {jettyOn ? t('uikit.jellyOn') : t('uikit.jellyOff')}
            </span>
          </div>
        </Card>

        <Card title={t('uikit.holdTitle')} desc={t('uikit.holdDesc')}>
          <ul className="uikit-rows">
            {DEMO_ROWS.map(rowId => (
              <li key={rowId} className={`uikit-row${removed.includes(rowId) ? ' uikit-row--removed' : ''}`}>
                <span className="uikit-row__name">{t('uikit.holdRow', { id: rowId.slice(-1).toUpperCase() })}</span>
                {removed.includes(rowId) ? (
                  <span className="uikit-row__gone">{t('uikit.holdRemoved')}</span>
                ) : (
                  <HoldToDeleteButton
                    label={t('uikit.holdLabel')}
                    hint={t('uikit.holdHint')}
                    doneLabel={t('uikit.holdDone')}
                    onConfirm={() => {
                      // 先让“对勾”状态可见，再移除条目，避免确认瞬间卸载组件
                      setTimeout(() => {
                        setRemoved(list => (list.includes(rowId) ? list : [...list, rowId]));
                      }, 500);
                    }}
                  />
                )}
              </li>
            ))}
          </ul>
          <div className="uikit-compact-row" data-compact-row="true">
            <span className="uikit-row__name">{t('uikit.holdCompactDemo')}</span>
            <HoldToDeleteButton
              size="compact"
              label={t('uikit.holdLabel')}
              hint={t('uikit.holdHint')}
              doneLabel={t('uikit.holdDone')}
              persist
            />
          </div>
          {removed.length > 0 && (
            <button type="button" className="uikit-restore" onClick={() => setRemoved([])}>
              {t('uikit.holdRestore')}
            </button>
          )}
        </Card>
      </div>
    </main>
  );
}
