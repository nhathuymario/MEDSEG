import { useState } from 'react';
import { api } from '../api/client';

export function useAnalysis() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const analyze = async (file, mode = 'pipeline') => {
    setLoading(true);
    setError(null);
    try {
      const fn = { detect: api.detect, segment: api.segment, pipeline: api.pipeline };
      const data = await fn[mode](file);
      const nextResult = { ...data, mode, timestamp: Date.now() };
      setResult(nextResult);
      return nextResult;
    } catch (e) {
      setError(e.message);
      return null;
    } finally {
      setLoading(false);
    }
  };

  return { analyze, result, loading, error, setResult };
}
