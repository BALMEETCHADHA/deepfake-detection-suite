const P = { fill: "none", stroke: "currentColor", strokeWidth: 1.6, strokeLinecap: "round", strokeLinejoin: "round" };
const wrap = (children) => ({ size = 48 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" {...P}>{children}</svg>
);
export const ImageIcon = wrap(<><rect x="3" y="4" width="18" height="16" rx="2" /><circle cx="9" cy="10" r="2" /><path d="M21 17l-5-5-9 8" /></>);
export const VideoIcon = wrap(<><rect x="2" y="5" width="14" height="14" rx="2" /><path d="M16 10l6-3v10l-6-3" /></>);
export const SignatureIcon = wrap(<><path d="M3 17c3-6 5-10 6-10s-2 8 0 8 3-5 5-5-1 5 1 5 3-2 6-3" /><path d="M3 21h18" /></>);
export const AudioIcon = wrap(<><path d="M4 10v4M8 6v12M12 3v18M16 7v10M20 10v4" /></>);
export const DocumentIcon = wrap(<><path d="M14 3H6a2 2 0 00-2 2v14a2 2 0 002 2h12a2 2 0 002-2V9z" /><path d="M14 3v6h6M8 13h8M8 17h6" /></>);
export const PdfIcon = wrap(<><path d="M14 3H6a2 2 0 00-2 2v14a2 2 0 002 2h12a2 2 0 002-2V9z" /><path d="M14 3v6h6" /><text x="12" y="17" fontSize="6" textAnchor="middle" fill="currentColor" stroke="none" fontFamily="sans-serif" fontWeight="700">PDF</text></>);
export const WordIcon = wrap(<><path d="M14 3H6a2 2 0 00-2 2v14a2 2 0 002 2h12a2 2 0 002-2V9z" /><path d="M14 3v6h6" /><text x="12" y="17" fontSize="6" textAnchor="middle" fill="currentColor" stroke="none" fontFamily="sans-serif" fontWeight="700">DOC</text></>);
export const PngIcon = wrap(<><path d="M14 3H6a2 2 0 00-2 2v14a2 2 0 002 2h12a2 2 0 002-2V9z" /><path d="M14 3v6h6" /><text x="12" y="17" fontSize="6" textAnchor="middle" fill="currentColor" stroke="none" fontFamily="sans-serif" fontWeight="700">IMG</text></>);
export const ScanIcon = wrap(<><path d="M4 8V5a1 1 0 011-1h3M16 4h3a1 1 0 011 1v3M20 16v3a1 1 0 01-1 1h-3M8 20H5a1 1 0 01-1-1v-3M3 12h18" /></>);
export const ICONS = { image: ImageIcon, video: VideoIcon, signature: SignatureIcon, audio: AudioIcon, document: DocumentIcon };
