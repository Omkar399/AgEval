import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8001';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 600000, // 10 minutes timeout for long-running evaluations
  headers: {
    'Content-Type': 'application/json',
  },
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

export interface EvaluationOverview {
  has_data: boolean;
  available_reports: string[];
  last_updated: string | null;
  evaluation_id?: string;
  status?: string;
  duration?: number;
  num_tasks?: number;
  num_metrics?: number;
  agent_info?: {
    name: string;
    model: string;
    provider: string;
  };
}

export interface PerformanceData {
  final_performance?: Record<string, number>;
  performance_summary?: any;
  aggregated_scores?: Record<string, Record<string, number>>;
  canonical_metrics?: Array<{
    name: string;
    definition: string;
    scale: string;
  }>;
}

export interface CalibrationData {
  bias_offsets?: Record<string, Record<string, number>>;
  agreement_analysis?: Record<string, any>;
  reliability_metrics?: Record<string, any>;
}

export interface AdaptiveData {
  has_adaptive_data: boolean;
  available_reports: string[];
  total_iterations?: number;
  convergence_achieved?: boolean;
  final_accuracy?: number;
  data: Record<string, any>;
}

export interface TasksData {
  tasks?: Array<{
    id: string;
    prompt: string;
    tier: string;
    category: string;
    description?: string;
    difficulty?: string;
    expected_response?: string;
    tags?: string[];
  }>;
  task_performance?: Record<string, {
    overall_score: number;
    metrics: Record<string, number>;
    judge_scores?: Record<string, number>;
    agent_output?: string;
  }>;
}

export interface JudgeData {
  judges: Array<{
    name: string;
    model: string;
    provider: string;
    config: Record<string, any>;
  }>;
  judge_metrics?: Record<string, {
    bias_offset: number;
    reliability: number;
    agreement_score: number;
    consistency: number;
    total_evaluations: number;
  }>;
  judge_comparisons?: Array<{
    judge1: string;
    judge2: string;
    agreement_rate: number;
    correlation: number;
    disagreement_patterns: string[];
  }>;
}

export interface LogEntry {
  timestamp: string;
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR' | 'SUCCESS';
  message: string;
  component?: string;
  evaluation_id?: string;
  phase?: string;
  judge?: string;
  task?: string;
}

export interface EvaluationProgress {
  phase: string;
  progress: number;
  current_task?: string;
  current_judge?: string;
  total_tasks?: number;
  completed_tasks?: number;
  estimated_remaining?: number;
}

export interface ExportData {
  metadata: {
    export_timestamp: string;
    ageval_version: string;
    export_options: any;
  };
  raw_scores?: Record<string, any>;
  calibration?: Record<string, any>;
  tasks?: Record<string, any>;
  agent_outputs?: Record<string, any>;
  judge_analysis?: Record<string, any>;
}

export interface BackupItem {
  id: string;
  name: string;
  timestamp: string;
  size: string;
  type: 'full' | 'evaluation' | 'config';
  description: string;
  status: 'completed' | 'in_progress' | 'failed';
}

export interface RadarChartData {
  metrics: string[];
  scores: number[];
  descriptions: string[];
}

export interface CalibrationChartData {
  bias_offsets: Record<string, Record<string, number>>;
  agreement_analysis: Record<string, any>;
  reliability_metrics: Record<string, any>;
}

export class ApiService {
  static async healthCheck(): Promise<{ status: string; timestamp: string }> {
    const response = await api.get('/api/health');
    return response.data;
  }

  static async getEvaluationOverview(): Promise<EvaluationOverview> {
    const response = await api.get('/api/evaluation/overview');
    return response.data;
  }

  static async getPerformanceData(): Promise<PerformanceData> {
    const response = await api.get('/api/evaluation/performance');
    return response.data;
  }

  static async getCalibrationData(): Promise<CalibrationData> {
    const response = await api.get('/api/evaluation/calibration');
    return response.data;
  }

  static async getTasksData(): Promise<TasksData> {
    const response = await api.get('/api/evaluation/tasks');
    return response.data;
  }

  static async getAdaptiveOverview(): Promise<AdaptiveData> {
    const response = await api.get('/api/adaptive/overview');
    return response.data;
  }

  static async getPerformanceRadarData(): Promise<RadarChartData> {
    const response = await api.get('/api/charts/performance-radar');
    return response.data;
  }

  static async getCalibrationChartData(): Promise<CalibrationChartData> {
    const response = await api.get('/api/charts/calibration-analysis');
    return response.data;
  }

  static async runEvaluation(config?: any): Promise<{ message: string; status: string }> {
    const response = await api.post('/api/evaluation/run', config);
    return response.data;
  }

  static async runAdaptiveEvaluation(config?: any): Promise<{ message: string; status: string }> {
    const response = await api.post('/api/evaluation/run-adaptive', config);
    return response.data;
  }

  static async getEvaluationStatus(): Promise<{ status: string }> {
    const response = await api.get('/api/evaluation/status');
    return response.data;
  }

  // New API methods for enhanced dashboard features

  static async getJudgeData(): Promise<JudgeData> {
    const response = await api.get('/api/judges/overview');
    return response.data;
  }

  static async getJudgeComparison(judge1: string, judge2: string): Promise<any> {
    const response = await api.get(`/api/judges/compare?judge1=${judge1}&judge2=${judge2}`);
    return response.data;
  }

  static async getTaskDetail(taskId: string): Promise<any> {
    const response = await api.get(`/api/tasks/${taskId}`);
    return response.data;
  }

  static async searchTasks(query: string, filters?: any): Promise<TasksData> {
    const params = new URLSearchParams({ query });
    if (filters) {
      Object.entries(filters).forEach(([key, value]) => {
        if (value) params.append(key, String(value));
      });
    }
    const response = await api.get(`/api/tasks/search?${params}`);
    return response.data;
  }

  static async getLogs(filters?: {
    level?: string;
    component?: string;
    limit?: number;
    since?: string;
  }): Promise<LogEntry[]> {
    const params = new URLSearchParams();
    if (filters) {
      Object.entries(filters).forEach(([key, value]) => {
        if (value) params.append(key, String(value));
      });
    }
    const response = await api.get(`/api/logs?${params}`);
    return response.data;
  }

  static async getEvaluationProgress(): Promise<EvaluationProgress | null> {
    try {
      const response = await api.get('/api/evaluation/progress');
      return response.data;
    } catch (error) {
      return null;
    }
  }

  static async exportData(options: {
    includeRawScores?: boolean;
    includeCalibrationData?: boolean;
    includeTaskDetails?: boolean;
    includeAgentOutputs?: boolean;
    includeJudgeAnalysis?: boolean;
    format?: 'json' | 'csv' | 'xlsx';
    compressionLevel?: 'none' | 'standard' | 'maximum';
  }): Promise<ExportData> {
    const response = await api.post('/api/data/export', options);
    return response.data;
  }

  static async importData(data: FormData): Promise<{ message: string; status: string }> {
    const response = await api.post('/api/data/import', data, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  static async getBackups(): Promise<BackupItem[]> {
    const response = await api.get('/api/data/backups');
    return response.data;
  }

  static async createBackup(name: string, type: 'full' | 'evaluation' | 'config' = 'full'): Promise<{ message: string; backup_id: string }> {
    const response = await api.post('/api/data/backups', { name, type });
    return response.data;
  }

  static async restoreBackup(backupId: string): Promise<{ message: string; status: string }> {
    const response = await api.post(`/api/data/backups/${backupId}/restore`);
    return response.data;
  }

  static async deleteBackup(backupId: string): Promise<{ message: string; status: string }> {
    const response = await api.delete(`/api/data/backups/${backupId}`);
    return response.data;
  }

  static async downloadBackup(backupId: string): Promise<Blob> {
    const response = await api.get(`/api/data/backups/${backupId}/download`, {
      responseType: 'blob',
    });
    return response.data;
  }

  static async getDataUsageStats(): Promise<{
    total_size: number;
    breakdown: Record<string, number>;
    recent_activity: Array<{
      action: string;
      timestamp: string;
      description: string;
    }>;
  }> {
    const response = await api.get('/api/data/usage');
    return response.data;
  }

  // WebSocket connection for real-time logs (to be implemented)
  static connectWebSocket(onMessage: (message: any) => void): WebSocket | null {
    try {
      const wsUrl = API_BASE_URL.replace('http', 'ws') + '/ws';
      const ws = new WebSocket(wsUrl);
      
      ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        onMessage(message);
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
      
      return ws;
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      return null;
    }
  }
}

export default api;