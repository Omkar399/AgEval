import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Alert,
  CircularProgress,
  Button,
  Chip,
  LinearProgress,
} from '@mui/material';
import { PlayArrow, Refresh, Assessment, Timer, Task } from '@mui/icons-material';
import { ApiService, EvaluationOverview } from '../services/api';

const Dashboard: React.FC = () => {
  const [overview, setOverview] = useState<EvaluationOverview | null>(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [runningType, setRunningType] = useState<'standard' | 'adaptive' | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await ApiService.getEvaluationOverview();
      setOverview(data);
    } catch (err) {
      setError('Failed to load evaluation data');
      console.error('Dashboard load error:', err);
    } finally {
      setLoading(false);
    }
  };

  const runEvaluation = async () => {
    try {
      setRunning(true);
      setError(null);
      
      // Call the synchronous evaluation endpoint
      // This will take 3-5 minutes to complete
      const result = await ApiService.runEvaluation();
      
      if (result.status === 'completed') {
        // Reload data to show new results
        await loadData();
        setRunning(false);
      } else if (result.status === 'failed') {
        setError(result.message || 'Evaluation failed');
        setRunning(false);
      }
    } catch (err: any) {
      setRunning(false);
      setError(err.response?.data?.message || 'Failed to run evaluation');
      console.error('Evaluation error:', err);
    }
  };

  const runAdaptiveEvaluation = async () => {
    try {
      setRunning(true);
      await ApiService.runAdaptiveEvaluation();
      // Poll for completion
      const checkStatus = async () => {
        const status = await ApiService.getEvaluationStatus();
        if (status.status === 'completed' || status.status === 'failed') {
          setRunning(false);
          loadData();
        } else {
          setTimeout(checkStatus, 2000);
        }
      };
      setTimeout(checkStatus, 1000);
    } catch (err) {
      setRunning(false);
      setError('Failed to start adaptive evaluation');
      console.error('Adaptive evaluation start error:', err);
    }
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 10000); // Refresh every 10 seconds
    return () => clearInterval(interval);
  }, []);

  if (loading && !overview) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h1" sx={{ 
          background: 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
        }}>
          AgEval Dashboard
        </Typography>
        <Box display="flex" gap={1}>
          <Button
            variant="contained"
            startIcon={running ? <CircularProgress size={20} color="inherit" /> : <PlayArrow />}
            onClick={runEvaluation}
            disabled={running}
            sx={{ 
              background: 'linear-gradient(45deg, #4CAF50 30%, #45a049 90%)',
              '&:hover': {
                background: 'linear-gradient(45deg, #45a049 30%, #4CAF50 90%)',
              }
            }}
          >
            {running ? 'Running...' : 'Run Evaluation'}
          </Button>
          <Button
            variant="contained"
            startIcon={running ? <CircularProgress size={20} color="inherit" /> : <PlayArrow />}
            onClick={runAdaptiveEvaluation}
            disabled={running}
            sx={{ 
              background: 'linear-gradient(45deg, #FF6B6B 30%, #4ECDC4 90%)',
              '&:hover': {
                background: 'linear-gradient(45deg, #4ECDC4 30%, #FF6B6B 90%)',
              }
            }}
          >
            {running ? 'Running...' : 'Run Adaptive'}
          </Button>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={loadData}
            disabled={loading}
          >
            Refresh
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {running && (
        <Alert severity="info" sx={{ mb: 2 }}>
          <Typography variant="body2" gutterBottom>
            <strong>Evaluation is running...</strong>
          </Typography>
          <Typography variant="body2" sx={{ mb: 1 }}>
            This process takes 3-5 minutes to complete:
          </Typography>
          <Typography variant="body2" component="div" sx={{ pl: 2 }}>
            • Running 9-phase standard evaluation<br/>
            • Running adaptive evaluation with IRT analysis<br/>
            • Evaluating with 3 judges (GPT-4, Claude, Gemini)<br/>
            • Processing 9 tasks across 5 metrics
          </Typography>
          <LinearProgress sx={{ mt: 2 }} />
        </Alert>
      )}

      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        {/* Status Cards Row */}
        <Box sx={{ 
          display: 'grid', 
          gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr', lg: 'repeat(4, 1fr)' }, 
          gap: 3 
        }}>
          {/* Status Overview */}
          <Card sx={{ 
            background: overview?.has_data 
              ? 'linear-gradient(135deg, #4CAF50 0%, #45a049 100%)'
              : 'linear-gradient(135deg, #ff9800 0%, #f57c00 100%)',
            color: 'white',
            height: '100%'
          }}>
            <CardContent>
              <Box display="flex" alignItems="center" mb={1}>
                <Assessment sx={{ mr: 1 }} />
                <Typography variant="h6">Status</Typography>
              </Box>
              <Typography variant="h4" fontWeight="bold">
                {overview?.has_data ? 'Ready' : 'No Data'}
              </Typography>
              <Typography variant="body2" sx={{ opacity: 0.9 }}>
                {overview?.status || 'Not evaluated'}
              </Typography>
            </CardContent>
          </Card>

          {/* Tasks Count */}
          <Card sx={{ 
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            color: 'white',
            height: '100%'
          }}>
            <CardContent>
              <Box display="flex" alignItems="center" mb={1}>
                <Task sx={{ mr: 1 }} />
                <Typography variant="h6">Tasks</Typography>
              </Box>
              <Typography variant="h4" fontWeight="bold">
                {overview?.num_tasks || 0}
              </Typography>
              <Typography variant="body2" sx={{ opacity: 0.9 }}>
                Evaluation tasks
              </Typography>
            </CardContent>
          </Card>

          {/* Metrics Count */}
          <Card sx={{ 
            background: 'linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%)',
            color: 'white',
            height: '100%'
          }}>
            <CardContent>
              <Box display="flex" alignItems="center" mb={1}>
                <Assessment sx={{ mr: 1 }} />
                <Typography variant="h6">Metrics</Typography>
              </Box>
              <Typography variant="h4" fontWeight="bold">
                {overview?.num_metrics || 0}
              </Typography>
              <Typography variant="body2" sx={{ opacity: 0.9 }}>
                Evaluation metrics
              </Typography>
            </CardContent>
          </Card>

          {/* Duration */}
          <Card sx={{ 
            background: 'linear-gradient(135deg, #A8E6CF 0%, #FF8B94 100%)',
            color: 'white',
            height: '100%'
          }}>
            <CardContent>
              <Box display="flex" alignItems="center" mb={1}>
                <Timer sx={{ mr: 1 }} />
                <Typography variant="h6">Duration</Typography>
              </Box>
              <Typography variant="h4" fontWeight="bold">
                {overview?.duration ? `${Math.round(overview.duration)}s` : 'N/A'}
              </Typography>
              <Typography variant="body2" sx={{ opacity: 0.9 }}>
                Last evaluation
              </Typography>
            </CardContent>
          </Card>
        </Box>

        {/* Info Cards Row */}
        <Box sx={{ 
          display: 'grid', 
          gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, 
          gap: 3 
        }}>
          {/* Agent Information */}
          {overview?.agent_info && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Agent Information
                </Typography>
                <Box display="flex" flexDirection="column" gap={1}>
                  <Box display="flex" alignItems="center" gap={1}>
                    <Typography variant="body2" color="textSecondary">Name:</Typography>
                    <Chip label={overview.agent_info.name} size="small" />
                  </Box>
                  <Box display="flex" alignItems="center" gap={1}>
                    <Typography variant="body2" color="textSecondary">Model:</Typography>
                    <Chip label={overview.agent_info.model} size="small" color="primary" />
                  </Box>
                  <Box display="flex" alignItems="center" gap={1}>
                    <Typography variant="body2" color="textSecondary">Provider:</Typography>
                    <Chip label={overview.agent_info.provider} size="small" color="secondary" />
                  </Box>
                </Box>
              </CardContent>
            </Card>
          )}

          {/* Available Reports */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Available Reports
              </Typography>
              <Box display="flex" flexWrap="wrap" gap={1}>
                {overview?.available_reports?.map((report) => (
                  <Chip 
                    key={report} 
                    label={report.replace(/_/g, ' ')} 
                    size="small" 
                    variant="outlined"
                  />
                )) || <Typography variant="body2" color="textSecondary">No reports available</Typography>}
              </Box>
              {overview?.last_updated && (
                <Typography variant="caption" color="textSecondary" sx={{ mt: 2, display: 'block' }}>
                  Last updated: {new Date(overview.last_updated).toLocaleString()}
                </Typography>
              )}
            </CardContent>
          </Card>
        </Box>
      </Box>

      {!overview?.has_data && (
        <Box mt={4}>
          <Alert severity="info">
            <Typography variant="h6" gutterBottom>
              Get Started
            </Typography>
            <Typography variant="body2" paragraph>
              No evaluation data found. Click "Run Evaluation" to start your first evaluation with the three-judge system.
            </Typography>
            <Typography variant="body2">
              The evaluation will analyze your agent across multiple tasks using GPT-4, Claude, and Gemini as judges.
            </Typography>
          </Alert>
        </Box>
      )}
    </Box>
  );
};

export default Dashboard;