// PanelError.jsx — Error display for a single comparison panel.
// Shows the endpoint URL and error details without affecting the other panel.

export default function PanelError({ error, onRetry }) {
  if (!error) return null;

  return (
    <div className="panel-error">
      <div className="panel-error-icon">⚠</div>
      <div className="panel-error-content">
        <div className="panel-error-title">Request failed</div>
        {error.endpoint && (
          <div className="panel-error-detail">
            <span className="panel-error-label">Endpoint:</span> {error.endpoint}
          </div>
        )}
        <div className="panel-error-detail">
          <span className="panel-error-label">Error:</span> {error.message}
        </div>
      </div>
      {onRetry && (
        <button className="panel-error-retry" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}
