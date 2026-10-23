import React, { useRef, useState } from 'react';
import C from '../../constants/colors';
import { getUploadedModelValidationErrors, getUploadedModelValidationStatus } from './uploadedModelValidation';
import './experimentCenter.css';

function getStatusColor(status) {
  if (status === 'valid') return C.green;
  if (status === 'pending' || status === 'timeout') return '#c89448';
  return '#d95c5c';
}

function getStatusLabel(status, labels) {
  if (status === 'valid') return labels.valid;
  if (status === 'pending') return labels.pending;
  if (status === 'timeout') return labels.timeout;
  return labels.invalid;
}

function ValidationMessages({ report, labels, fieldHintStyle }) {
  const errors = getUploadedModelValidationErrors(report, labels.timeout);
  const warnings = Array.isArray(report?.warnings) ? report.warnings : [];

  if (errors.length === 0 && warnings.length === 0) {
    return <div style={{ ...fieldHintStyle, marginTop: 0, color: C.green }}>{labels.ready}</div>;
  }

  return (
    <div style={{ display: 'grid', gap: 6 }}>
      {errors.map((message) => (
        <div key={`error-${message}`} style={{ ...fieldHintStyle, marginTop: 0, color: '#d95c5c' }}>
          {message}
        </div>
      ))}
      {warnings.map((message) => (
        <div key={`warning-${message}`} style={{ ...fieldHintStyle, marginTop: 0, color: '#c89448' }}>
          {message}
        </div>
      ))}
    </div>
  );
}

function ModelRenameForm({ model, onRename, onCancel, busy, labels }) {
  const [name, setName] = useState(model.display_name || '');
  const [error, setError] = useState('');

  return (
    <form
      className="experiment-uploaded-rename"
      onSubmit={async (event) => {
        event.preventDefault();
        if (busy) return;
        const normalized = name.trim();
        if (!normalized || Array.from(normalized).length > 120) {
          setError(labels.nameRequired);
          return;
        }
        if (await onRename(model.id, normalized)) onCancel();
      }}
    >
      <label>
        <span>{labels.name}</span>
        <input
          type="text"
          value={name}
          maxLength={120}
          autoFocus
          disabled={busy}
          aria-invalid={Boolean(error)}
          onChange={(event) => { setName(event.target.value); setError(''); }}
          onKeyDown={(event) => {
            if (event.key === 'Escape' && !busy) { event.stopPropagation(); onCancel(); }
          }}
        />
      </label>
      <div className="experiment-uploaded-detail-actions">
        <button type="submit" className="experiment-expert-action" disabled={busy}>{labels.saveName}</button>
        <button type="button" className="experiment-expert-action" disabled={busy} onClick={onCancel}>{labels.cancelRename}</button>
      </div>
      {error ? <div className="experiment-uploaded-model-error" role="alert">{error}</div> : null}
    </form>
  );
}

/**
 * 上传模型区域（紧凑版）。
 *
 * 默认只显示当前模型的胶囊式摘要（截断显示名称、版本 · 自定义参数数量）
 * 与主上传动作（上传模型）。整套模型管理（列表、下载、改名、重新校验、删除）
 * 收在可展开的「管理模型」里，不抢占主上传操作的视觉层级。
 * 自定义模型参数的编辑入口只保留在右侧配置检查器，这里不再重复提供。
 *
 * 上传、选择、下载源码、改名、重新校验、删除、模板与说明下载、区内 inline error
 * 全部保留；校验失败、版本不匹配等错误通过 `inlineError` 直接显示在区域顶部。
 * 展开区里的「文件格式要求」与底部「校验状态」两行已按用户要求删除。
 */
export default function UploadedModelPanel({
  models = [],
  selectedId,
  onSelect,
  onUpload,
  onRevalidate,
  onDelete,
  onRename,
  onDownload,
  uploading = false,
  busy = false,
  selectionDisabled = false,
  guideDownloadUrl,
  templateDownloadUrl,
  sharedGateTemplateDownloadUrl,
  inlineError = '',
  statusLabel = '',
  statusTone = 'ok',
  labels,
  sectionTitleStyle,
  fieldHintStyle,
}) {
  const fileRef = useRef(null);
  const [manageOpen, setManageOpen] = useState(false);
  const [renamingId, setRenamingId] = useState(null);
  const selected = models.find((item) => item.id === selectedId) || null;
  const selectedStatus = getUploadedModelValidationStatus(selected);
  const paramCount = Object.keys(selected?.param_schema || {}).length;
  const selectedName = selected?.display_name || selected?.original_filename || labels.noFilename;

  const pickFile = () => fileRef.current?.click();

  return (
    <div className="experiment-uploaded-model" data-uploaded-model-panel="true" data-has-selection={selected ? 'true' : 'false'} data-model-state={selected ? (selected.validation_status === 'valid' ? 'valid' : 'invalid') : 'empty'}>
      <input
        ref={fileRef}
        type="file"
        accept=".py"
        hidden
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) onUpload(file);
          event.target.value = '';
        }}
      />

      {/* 错误、校验失败与版本不匹配都在上传模型区域内解释。 */}
      {inlineError ? (
        <div className="experiment-uploaded-model-error" role="alert" data-uploaded-model-error="true">
          {inlineError}
        </div>
      ) : null}

      {selected ? (
        <div className="experiment-uploaded-card" data-uploaded-model-card="true" data-validation-status={selectedStatus}>
          <div className="experiment-uploaded-body">
            <div className="experiment-uploaded-name" title={selectedName}>{selectedName}</div>
            <div className="experiment-uploaded-meta">
              {`v${selected.version ?? '--'} · ${labels.summaryParamCount ? labels.summaryParamCount(paramCount) : `${paramCount}`}`}
            </div>
          </div>
          <span className="experiment-uploaded-status" data-tone={statusTone} data-validation-status={selectedStatus} data-uploaded-model-status="true">
            {statusLabel || getStatusLabel(selectedStatus, labels)}
          </span>
        </div>
      ) : (
        <div className="experiment-uploaded-empty" data-uploaded-model-empty="true">
          <strong>{labels.missing}</strong>
          <span>{labels.hint}</span>
        </div>
      )}

      <div className="experiment-uploaded-actions">
        <button
          type="button"
          className="experiment-uploaded-upload"
          data-uploaded-model-upload="true"
          onClick={pickFile}
          disabled={uploading || busy}
        >
          {uploading ? labels.uploading : (selected ? labels.uploadAgain : labels.upload)}
        </button>
        <button
          type="button"
          className="experiment-uploaded-link is-quiet"
          aria-expanded={manageOpen}
          aria-controls="experiment-uploaded-manage"
          onClick={() => { setManageOpen((value) => !value); setRenamingId(null); }}
        >
          {labels.manage}
        </button>
        {guideDownloadUrl ? (
          <a href={guideDownloadUrl} download className="experiment-uploaded-link is-quiet">
            {labels.downloadGuide}
          </a>
        ) : null}
        {templateDownloadUrl ? (
          <a href={templateDownloadUrl} download className="experiment-uploaded-link is-quiet">
            {labels.downloadTemplate}
          </a>
        ) : null}
        {sharedGateTemplateDownloadUrl ? (
          <a href={sharedGateTemplateDownloadUrl} download className="experiment-uploaded-link is-quiet">
            {labels.downloadSharedGateTemplate}
          </a>
        ) : null}
      </div>

      {manageOpen ? (
        <div className="experiment-uploaded-manage" id="experiment-uploaded-manage" data-uploaded-model-manage="true">
          {models.length === 0 ? (
            <div className="experiment-uploaded-empty" role="status">
              <strong>{labels.missing}</strong>
              <span>{labels.hint}</span>
            </div>
          ) : (
            <div className="experiment-uploaded-list" role="list">
              {models.map((model) => {
                const active = selectedId === model.id;
                const validationStatus = getUploadedModelValidationStatus(model);
                return (
                  <button
                    key={model.id}
                    type="button"
                    role="listitem"
                    className="experiment-uploaded-item"
                    aria-pressed={active}
                    data-uploaded-model-item={model.id}
                    onClick={() => { onSelect(model.id); setRenamingId(null); }}
                    disabled={selectionDisabled}
                  >
                    <span className="experiment-uploaded-item-name" title={model.original_filename || ''}>
                      {model.display_name || model.original_filename || labels.unnamed}
                    </span>
                    <span className="experiment-uploaded-item-status" style={{ color: getStatusColor(validationStatus) }}>
                      {getStatusLabel(validationStatus, labels)}
                    </span>
                    <span className="experiment-uploaded-item-meta">
                      {`v${model.version ?? '--'} / ${model.original_filename || labels.noFilename}`}
                    </span>
                  </button>
                );
              })}
            </div>
          )}

          {selected ? (
            <div className="experiment-uploaded-detail">
              <div className="experiment-uploaded-detail-actions">
                <button
                  type="button"
                  className="experiment-expert-action"
                  onClick={() => onDownload(selected.id)}
                  disabled={busy}
                >
                  {labels.download}
                </button>
                <button
                  type="button"
                  className="experiment-expert-action"
                  onClick={() => setRenamingId(selected.id)}
                  disabled={busy}
                >
                  {labels.rename}
                </button>
                <button
                  type="button"
                  className="experiment-expert-action"
                  onClick={() => onRevalidate(selected.id)}
                  disabled={busy}
                >
                  {labels.revalidate}
                </button>
                <button
                  type="button"
                  className="experiment-expert-action"
                  onClick={pickFile}
                  disabled={uploading || busy}
                >
                  {labels.replace}
                </button>
                <button
                  type="button"
                  className="experiment-expert-action is-danger"
                  onClick={() => onDelete(selected.id)}
                  disabled={busy}
                >
                  {labels.delete}
                </button>
              </div>
              {renamingId === selected.id ? (
                <ModelRenameForm
                  key={`${selected.id}:${selected.display_name}`}
                  model={selected}
                  onRename={onRename}
                  onCancel={() => setRenamingId(null)}
                  busy={busy}
                  labels={labels}
                />
              ) : null}
              <ValidationMessages report={selected.validation_report} labels={labels} fieldHintStyle={fieldHintStyle} />
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
