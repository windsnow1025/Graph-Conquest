import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    rules: {
      // The core is reached through its API and its configuration data
      'no-restricted-imports': ['error', {
        patterns: [{group: ['**/lib/*', '!**/lib/GameAPI', '!**/lib/data'], message: 'Import the core through lib/GameAPI, its configuration data through lib/data.'}],
      }],
    },
  },
  {
    // The conformance test compares the API with the engine
    files: ['training/scripts/test/testAPI.ts'],
    rules: {'no-restricted-imports': 'off'},
  },
])
