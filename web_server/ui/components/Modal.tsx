'use client';

import { useEffect, type ReactNode } from 'react';
import { createPortal } from 'react-dom';

// Lightweight modal: Esc / backdrop click to close, scroll-locked, no deps.
// Rendered through a portal to <body> so its fixed-position overlay always
// covers the full viewport, regardless of any transformed/positioned ancestor
// (a transform/filter/contain ancestor would otherwise trap `fixed` into that
// ancestor's box and leave part of the page undarkened).
export function Modal({
  title,
  onClose,
  children,
}: {
  title: ReactNode;
  onClose: () => void;
  children: ReactNode;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.removeEventListener('keydown', onKey);
      document.body.style.overflow = prev;
    };
  }, [onClose]);

  if (typeof document === 'undefined') return null;

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/40 motion-safe:animate-fade-in" onClick={onClose} aria-hidden />
      <div
        role="dialog"
        aria-modal="true"
        className="relative z-10 w-full max-w-2xl max-h-[80vh] flex flex-col rounded-2xl border border-gray-200 bg-white shadow-xl"
      >
        <div className="flex items-start justify-between gap-3 px-5 py-3 border-b border-gray-100">
          <div className="min-w-0 text-sm font-medium text-gray-900">{title}</div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="shrink-0 -mr-1 text-gray-400 hover:text-gray-700"
          >
            ✕
          </button>
        </div>
        <div className="overflow-y-auto px-5 py-4">{children}</div>
      </div>
    </div>,
    document.body,
  );
}
