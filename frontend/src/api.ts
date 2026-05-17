import type {
  DatasetSelectionResponse,
  EvaluationJob,
  EvaluationRequest,
  MetaResponse
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

function fallbackApiBase() {
  if (API_BASE || typeof window === "undefined") {
    return null;
  }

  const { protocol, hostname } = window.location;
  return `${protocol}//${hostname}:7860`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const fetchJson = async (base: string) => {
    const headers: Record<string, string> = {};
    if (init?.headers instanceof Headers) {
      init.headers.forEach((value, key) => {
        headers[key] = value;
      });
    } else if (Array.isArray(init?.headers)) {
      for (const [key, value] of init.headers) {
        headers[key] = value;
      }
    } else if (init?.headers) {
      Object.assign(headers, init.headers);
    }
    if (init?.body !== undefined && !("Content-Type" in headers)) {
      headers["Content-Type"] = "application/json";
    }

    const response = await fetch(`${base}${path}`, {
      headers,
      ...init
    });

    if (!response.ok) {
      let message = response.statusText;
      try {
        const body = await response.json();
        message = body.detail ?? message;
      } catch {
        // Keep the HTTP status text when the server did not return JSON.
      }
      throw new Error(message);
    }

    return response.json() as Promise<T>;
  };

  try {
    return await fetchJson(API_BASE);
  } catch (error) {
    const fallback = fallbackApiBase();
    if (!fallback) {
      throw error;
    }

    try {
      return await fetchJson(fallback);
    } catch {
      throw error;
    }
  }
}

export function fetchMeta() {
  return request<MetaResponse>("/api/meta");
}

export function registerModel(alias: string, modelPath: string) {
  return request<{ status: string; models: MetaResponse["models"]; selected: string }>(
    "/api/models/register",
    {
      method: "POST",
      body: JSON.stringify({ alias, model_path: modelPath })
    }
  );
}

export function fetchDatasetSelection(datasetNames: string[]) {
  return request<DatasetSelectionResponse>("/api/datasets/selection", {
    method: "POST",
    body: JSON.stringify(datasetNames)
  });
}

export function startEvaluation(payload: EvaluationRequest) {
  return request<EvaluationJob>("/api/evaluations", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function fetchEvaluation(jobId: string) {
  return request<EvaluationJob>(`/api/evaluations/${jobId}`);
}
