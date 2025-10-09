'use client';

import { useTheme } from '@/components/ThemeProvider';

export default function TestThemePage() {
  const { theme, toggleTheme } = useTheme();

  return (
    <div className="min-h-screen p-8">
      <div className="max-w-4xl mx-auto space-y-8">
        <h1 className="text-4xl font-bold text-gray-900">
          Theme Test Page
        </h1>

        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h2 className="text-2xl font-bold mb-4 text-gray-900">
            Current Theme: {theme}
          </h2>
          <button
            onClick={toggleTheme}
            className="px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
          >
            Toggle Theme (Current: {theme})
          </button>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-100 p-4 rounded">
            <p className="text-gray-900">Background Test</p>
          </div>
          <div className="bg-white p-4 rounded border border-gray-300">
            <p className="text-gray-900">Border Test</p>
          </div>
        </div>

        <div className="space-y-4">
          <p className="text-gray-900">
            This text should be dark in light mode and white in dark mode
          </p>
          <p className="text-gray-600">
            This text should be gray-600 in light mode and gray-400 in dark mode
          </p>
        </div>

        <div className="bg-red-100 p-4 rounded">
          <p className="text-red-900">
            Red background test
          </p>
        </div>

        <div className="bg-blue-100 p-4 rounded">
          <p className="text-blue-900">
            Blue background test
          </p>
        </div>

        <div className="mt-8 p-4 bg-yellow-100 rounded">
          <h3 className="font-bold text-yellow-900 mb-2">
            Debug Info
          </h3>
          <pre className="text-sm text-yellow-800">
            Theme State: {theme}
            {'\n'}
            LocalStorage: {typeof window !== 'undefined' ? localStorage.getItem('theme') : 'N/A'}
            {'\n'}
            HTML has dark class: {typeof document !== 'undefined' ? document.documentElement.classList.contains('dark').toString() : 'N/A'}
          </pre>
        </div>
      </div>
    </div>
  );
}

