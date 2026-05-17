export type ModelOption = {
  label: string;
  value: string;
  path: string;
};

export type DatasetOption = {
  name: string;
  data_type: string;
  metrics: string[];
  description: string;
};

export type MetaResponse = {
  models: ModelOption[];
  datasets: DatasetOption[];
  default_datasets: string[];
  default_metrics: string[];
  default_dataset_info: string;
  available_gpu_count: number;
  gpu_choices: string[];
  default_gpu_choice: string;
  backend: string;
};

export type DatasetSelectionResponse = {
  metrics: string[];
  dataset_info: string;
};

export type EvaluationRequest = {
  model_name?: string | null;
  custom_model?: string | null;
  dataset_names: string[];
  selected_metrics: string[];
  tp_size_choice?: string | number | null;
  tp_size_input?: string | number | null;
};

export type EvaluationJob = {
  id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  created_at: string;
  updated_at: string;
  request: EvaluationRequest;
  result?: Record<string, unknown> | null;
  error?: string | null;
};
