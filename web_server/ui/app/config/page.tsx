'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Check, Trash2, Copy } from 'lucide-react';
import { api, type ModelChoice } from '@/lib/api';
import {
  getStoredKey,
  setStoredKey,
  clearStoredKey,
  getStoredModel,
  setStoredModel,
} from '@/lib/public-mode';

// Config page: enter the OpenAI API key (saved to a browser cookie so it
// persists across tabs and reloads), pick the synthesis-model override (also a
// browser cookie), and recover past runs after a cookie reset via a recovery
// code (the browser's caesar_id identity).
export default function ConfigPage() {
  const router = useRouter();
  const [value, setValue] = useState('');
  const [hasKey, setHasKey] = useState(false);
  const [saved, setSaved] = useState(false);
  // Synthesis-model override. '' = use the preset's default model.
  const [models, setModels] = useState<ModelChoice[]>([]);
  const [model, setModel] = useState('');
  const [hasModel, setHasModel] = useState(false);
  const [modelSaved, setModelSaved] = useState(false);

  // Recovery code = this browser's caesar_id (fetched from the server, which
  // owns the HttpOnly cookie). Restoring sets that cookie to a pasted code.
  const [recoveryCode, setRecoveryCode] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [restoreInput, setRestoreInput] = useState('');
  const [restoring, setRestoring] = useState(false);
  const [restoreError, setRestoreError] = useState<string | null>(null);

  useEffect(() => {
    const existing = getStoredKey();
    setValue(existing);
    setHasKey(existing.length > 0);
    const storedModel = getStoredModel();
    setModel(storedModel);
    setHasModel(storedModel.length > 0);
    api.getModels().then(setModels).catch(() => setModels([]));
    fetch('/api/whoami', { cache: 'no-store' })
      .then((r) => r.json())
      .then((d) => setRecoveryCode(typeof d?.owner_id === 'string' ? d.owner_id : null))
      .catch(() => setRecoveryCode(null));
  }, []);

  const onSave = () => {
    const v = value.trim();
    setStoredKey(v);
    setHasKey(v.length > 0);
    setSaved(true);
    // Brief "Saved" confirmation, then return to the main query page.
    window.setTimeout(() => router.push('/'), 600);
  };

  const onClear = () => {
    clearStoredKey();
    setValue('');
    setHasKey(false);
    setSaved(false);
  };

  const onModelSave = () => {
    setStoredModel(model);
    setHasModel(model.length > 0);
    setModelSaved(true);
    // Brief "Saved" confirmation; stay on the page (unlike the key save).
    window.setTimeout(() => setModelSaved(false), 1500);
  };

  const onCopy = async () => {
    if (!recoveryCode) return;
    try {
      await navigator.clipboard.writeText(recoveryCode);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      // clipboard may be unavailable; ignore.
    }
  };

  const onRestore = async () => {
    const code = restoreInput.trim();
    if (!code) return;
    setRestoring(true);
    setRestoreError(null);
    try {
      const res = await fetch('/api/restore', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.detail ?? `Restore failed (${res.status})`);
      }
      // Cookie is now set to the recovered identity; go to the run list.
      window.location.href = '/runs';
    } catch (e) {
      setRestoreError((e as Error).message);
      setRestoring(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto px-4 sm:px-6 py-10 space-y-6">
      <div className="rounded-2xl border border-gray-200 bg-white shadow-sm p-6">
        <div className="flex items-center gap-2 mb-1">
          <h1 className="text-lg font-semibold text-gray-900">OpenAI API key</h1>
          <span
            className={`ml-1 inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full ${
              hasKey ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'
            }`}
          >
            {hasKey ? 'Key saved' : 'No key saved'}
          </span>
        </div>
        <p className="text-sm text-gray-500 mb-5">
          Caesar runs on your own OpenAI key. It is stored only in your browser (a
          cookie on this site), sent only to run your queries, and never saved on the
          server.
        </p>

        <label htmlFor="api-key" className="block text-sm font-medium text-gray-700 mb-1">
          Key
        </label>
        {/* NOT type=password: Chrome's built-in manager autofills the saved
            site password (e.g. the admin-login password) into ANY password
            field, ignoring autocomplete/data-*-ignore hints, which silently
            overwrote the saved key. Render as a text field masked via CSS
            (-webkit-text-security) so Chrome never treats it as a login
            password, while the value still displays as dots. */}
        <input
          id="api-key"
          name="caesar-openai-key"
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="sk-..."
          autoComplete="off"
          data-1p-ignore="true"
          data-lpignore="true"
          data-bwignore="true"
          data-form-type="other"
          spellCheck={false}
          className="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-brand-200 focus:border-brand-500 [-webkit-text-security:disc]"
        />

        <div className="mt-4 flex items-center gap-2">
          <button
            type="button"
            onClick={onSave}
            disabled={value.trim().length === 0}
            className="inline-flex items-center gap-1.5 rounded-xl bg-brand-800 hover:bg-brand-900 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-medium px-4 py-2 transition-colors"
          >
            {saved ? (
              <>
                <Check size={16} /> Saved
              </>
            ) : (
              'Save key'
            )}
          </button>
          {hasKey && (
            <button
              type="button"
              onClick={onClear}
              className="inline-flex items-center gap-1.5 rounded-xl border border-gray-200 text-gray-600 hover:bg-gray-50 text-sm font-medium px-3 py-2 transition-colors"
            >
              <Trash2 size={15} /> Clear
            </button>
          )}
        </div>
      </div>

      <div className="rounded-2xl border border-gray-200 bg-white shadow-sm p-6">
        <div className="flex items-center gap-2 mb-1">
          <h2 className="text-lg font-semibold text-gray-900">Synthesis model</h2>
          <span
            className={`ml-1 inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full ${
              hasModel ? 'bg-green-50 text-green-700' : 'bg-gray-100 text-gray-500'
            }`}
          >
            {hasModel ? 'Custom model saved' : 'Preset default'}
          </span>
        </div>
        <p className="text-sm text-gray-500 mb-5">
          The model that writes the final answer (synthesis). Exploration and
          retrieval keep the preset&apos;s model. Your choice is saved in this
          browser and applied to new runs.
        </p>
        <label htmlFor="synthesis-model" className="block text-sm font-medium text-gray-700 mb-1">
          Model
        </label>
        <select
          id="synthesis-model"
          value={model}
          onChange={(e) => {
            setModel(e.target.value);
            setModelSaved(false);
          }}
          className="w-full rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm text-gray-700 focus:outline-none focus:ring-2 focus:ring-brand-200 focus:border-brand-500"
        >
          <option value="">Preset default</option>
          {models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.id}
              {m.input_per_mtok != null && m.output_per_mtok != null
                ? ` ($${m.input_per_mtok}/$${m.output_per_mtok} per 1M)`
                : ''}
            </option>
          ))}
        </select>

        <div className="mt-4">
          <button
            type="button"
            onClick={onModelSave}
            className="inline-flex items-center gap-1.5 rounded-xl bg-brand-800 hover:bg-brand-900 text-white text-sm font-medium px-4 py-2 transition-colors"
          >
            {modelSaved ? (
              <>
                <Check size={16} /> Saved
              </>
            ) : (
              'Save model'
            )}
          </button>
        </div>
      </div>

      <div className="rounded-2xl border border-gray-200 bg-white shadow-sm p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-1">Recover past runs</h2>
        <p className="text-sm text-gray-500 mb-5">
          Your runs are tied to this browser. Save your recovery code; if you clear
          cookies or switch browsers, paste it back to get your runs again. Anyone with
          this code can view your runs, so keep it private.
        </p>

        {/* A real form with a hidden username + password-autocomplete fields so
            browser password managers (1Password, Chrome, etc.) capture the
            recovery code and autofill it on a new browser/device. */}
        <form onSubmit={(e) => { e.preventDefault(); onRestore(); }}>
          <input
            type="text"
            name="username"
            value="caesar-recovery"
            autoComplete="username"
            readOnly
            aria-hidden="true"
            tabIndex={-1}
            className="hidden"
          />

          <label htmlFor="recovery-code" className="block text-sm font-medium text-gray-700 mb-1">
            Your recovery code
          </label>
          <div className="flex items-center gap-2">
            <input
              id="recovery-code"
              type="text"
              value={recoveryCode ?? ''}
              readOnly
              autoComplete="new-password"
              placeholder="..."
              className="flex-1 rounded-lg border border-gray-200 bg-gray-50 px-3 py-2 text-sm text-gray-800 font-mono break-all focus:outline-none"
            />
            <button
              type="button"
              onClick={onCopy}
              disabled={!recoveryCode}
              className="inline-flex items-center gap-1.5 rounded-xl border border-gray-200 text-gray-600 hover:bg-gray-50 disabled:opacity-50 text-sm font-medium px-3 py-2 transition-colors"
            >
              {copied ? <Check size={15} /> : <Copy size={15} />}
              {copied ? 'Copied' : 'Copy'}
            </button>
          </div>

          <label
            htmlFor="restore-code"
            className="block text-sm font-medium text-gray-700 mb-1 mt-5"
          >
            Restore a previous session
          </label>
          <div className="flex items-center gap-2">
            <input
              id="restore-code"
              type="password"
              value={restoreInput}
              onChange={(e) => setRestoreInput(e.target.value)}
              placeholder="Paste a recovery code"
              autoComplete="current-password"
              spellCheck={false}
              className="flex-1 rounded-lg border border-gray-200 px-3 py-2 text-sm placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-brand-200 focus:border-brand-500"
            />
            <button
              type="submit"
              disabled={restoring || restoreInput.trim().length === 0}
              className="inline-flex items-center rounded-xl bg-brand-800 hover:bg-brand-900 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-medium px-4 py-2 transition-colors"
            >
              {restoring ? 'Restoring...' : 'Restore'}
            </button>
          </div>
          {restoreError && (
            <p className="mt-2 text-sm text-red-600" role="alert">
              {restoreError}
            </p>
          )}
        </form>
      </div>
    </div>
  );
}
