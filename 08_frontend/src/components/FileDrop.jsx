export default function FileDrop({ label, optional, accept, file, onChange, id }) {
  return (
    <div className="field">
      <label className="field-label" htmlFor={id}>
        {label} {optional && <span className="optional">(optional)</span>}
      </label>
      <label className={`dropzone ${file ? "has-file" : ""}`} htmlFor={id}>
        <input
          id={id}
          type="file"
          accept={accept}
          onChange={(e) => onChange(e.target.files?.[0] || null)}
        />
        {file ? (
          <div className="dropzone-filename">{file.name}</div>
        ) : (
          <div className="dropzone-hint">Click to choose a file{accept ? ` (${accept})` : ""}</div>
        )}
      </label>
    </div>
  );
}
