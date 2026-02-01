export enum JobStatus {
  QUEUED = 'QUEUED',
  RUNNING = 'RUNNING',
  SUCCEEDED = 'SUCCEEDED',
  FAILED = 'FAILED',
  CANCELED = 'CANCELED',
}

export type SamplingMode =
  | 'Unconditional'
  | 'Pharmacophore-conditioned'
  | 'Protein-conditioned'
  | 'Protein+Pharmacophore-conditioned'
  | 'Rigid Docking'
  | 'Rigid Docking + Pharmacophore';

export interface SamplingParams {
  sampling_mode?: SamplingMode;
  docking_mode?: DockingMode;
  seed?: number | null;
  n_samples: number;
  steps: number;
  device?: string;
  n_lig_atoms_mean?: number | null;
  n_lig_atoms_std?: number | null;
  pocket_selection?: {
    type: 'center' | 'residues' | 'file';
    value: [number, number, number] | Array<{ chain: string; res_id: number }> | string;
    bbox_length?: number;
  };
}

export interface JobSubmission {
  params: SamplingParams;
  uploads: string[];
  num_samples?: number;
  job_id?: string;
}

export interface JobResponse {
  job_id: string;
  message?: string;
}

export interface JobStatusResponse {
  job_id: string;
  state: JobStatus;
  progress: number;
  message?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  elapsed_seconds?: number | null;
}

export interface ArtifactInfo {
  id: string;
  filename: string;
  format: string;
  size: number;
  path_or_url: string;
  sha256?: string;
}

export interface JobResultResponse {
  job_id: string;
  state: JobStatus;
  artifacts: ArtifactInfo[];
  logs_url?: string;
  params: SamplingParams;
  elapsed_seconds?: number | null;
  error_message?: string | null;
}

export interface UploadInitResponse {
  upload_token: string;
  upload_url?: string;
  max_file_size: number;
}

export interface PharmacophoreFeature {
  index: number;
  type: string;
  position: [number, number, number];
  color: string;
  selected: boolean;
}

export interface PharmacophoreExtractionResult {
  pharmacophores: Array<{
    type: string;
    position: [number, number, number];
  }>;
  n_pharmacophores: number;
}

export interface PharmacophoreToXyzResult {
  xyz_content: string;
  n_selected: number;
  n_total: number;
}

export interface JobMetadata {
  job_id: string;
  rq_job_id?: string;
  params: SamplingParams;
  input_files?: Array<{
    filename: string;
    path: string;
    size: number;
    sha256: string;
  }>;
  created_at: string;
  state?: JobStatus;
  progress?: number;
  elapsed_seconds?: number;
  message?: string;
}

export interface JobsListResponse {
  jobs: JobMetadata[];
}

// Docking types
export type DockingMode = 'Rigid Docking' | 'Rigid Docking + Pharmacophore';

export interface PocketInfo {
  id: string;
  center: [number, number, number];
  bbox_length: number;
  residues?: Array<{ chain: string; res_id: number }>;
  score?: number;
}

export interface DockingParams {
  docking_mode: DockingMode;
  seed?: number | null;
  n_samples: number;
  steps: number;
  device?: string;
  pocket_selection?: {
    type: 'center' | 'residues' | 'file';
    value: [number, number, number] | Array<{ chain: string; res_id: number }> | string;
    bbox_length?: number;
  };
}

export interface DockingJobSubmission {
  params: DockingParams;
  uploads: string[];
  job_id?: string;
}

export interface PocketDetectionResponse {
  pockets: PocketInfo[];
}


