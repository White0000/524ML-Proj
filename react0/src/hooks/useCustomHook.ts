import { useState, useEffect, useCallback } from 'react'

interface UseCustomHookOptions {
  defaultValue?: string
  syncLocalStorage?: boolean
  localStorageKey?: string
}

export function useCustomHook(options?: UseCustomHookOptions) {
  const { defaultValue = 'initial', syncLocalStorage = false, localStorageKey = 'customHookValue' } = options || {}
  const [value, setValue] = useState(() => {
    if (!syncLocalStorage) return defaultValue
    try {
      const stored = localStorage.getItem(localStorageKey)
      return stored !== null ? stored : defaultValue
    } catch {
      return defaultValue
    }
  })

  const updateValue = useCallback((newVal: string) => {
    setValue(newVal)
  }, [])

  useEffect(() => {
    if (syncLocalStorage) {
      try {
        localStorage.setItem(localStorageKey, value)
      } catch {}
    }
  }, [value, syncLocalStorage, localStorageKey])

  return { value, setValue: updateValue }
}
