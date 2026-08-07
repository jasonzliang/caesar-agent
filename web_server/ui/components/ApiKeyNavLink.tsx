'use client';

import { useEffect, useState } from 'react';
import { getStoredKey } from '@/lib/public-mode';

// Nav-bar link to the dedicated /config page. The link text turns red when no
// key is saved in this browser (and matches the rest of the nav otherwise), so
// the status is conveyed by color alone without an icon or indicator dot.
export function ApiKeyNavLink() {
  const [hasKey, setHasKey] = useState<boolean | null>(null);

  useEffect(() => {
    setHasKey(getStoredKey().length > 0);
  }, []);

  return (
    <a
      href="/config"
      suppressHydrationWarning
      className={hasKey === false ? 'text-red-600 hover:text-red-700' : 'hover:text-gray-900'}
    >
      Config
    </a>
  );
}
