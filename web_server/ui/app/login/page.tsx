import { Suspense } from 'react';
import { LoginForm } from './LoginForm';

export const metadata = { title: 'Sign in · Caesar' };

export default function LoginPage() {
  return (
    <div className="min-h-[calc(100vh-8rem)] flex items-center justify-center px-4">
      <div className="w-full max-w-sm">
        <div className="text-center mb-8 space-y-2">
          <h1 className="text-2xl font-semibold tracking-tight text-gray-900">
            Caesar Web Demo
          </h1>
          <p className="text-sm text-gray-500">
            Enter the access password to continue.
          </p>
        </div>
        <Suspense fallback={null}>
          <LoginForm />
        </Suspense>
      </div>
    </div>
  );
}
