import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Alert,
  CircularProgress,
  Chip,
} from '@mui/material';
import { ApiService, AdaptiveData } from '../services/api';

const AdaptiveAnalysis: React.FC = () => {
  const [adaptiveData, setAdaptiveData] = useState<AdaptiveData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const data = await ApiService.getAdaptiveOverview();
      setAdaptiveData(data);
    } catch (err) {
      setError('Failed to load adaptive analysis data');
      console.error('Adaptive analysis load error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ m: 2 }}>
        {error}
      </Alert>
    );
  }

  if (!adaptiveData?.has_adaptive_data) {
    return (
      <Alert severity="info" sx={{ m: 2 }}>
        <Typography variant="h6" gutterBottom>
          Adaptive Analysis Not Available
        </Typography>
        <Typography variant="body2" paragraph>
          No adaptive evaluation data found. This feature requires running the enhanced adaptive evaluation pipeline.
        </Typography>
        <Typography variant="body2">
          The adaptive system uses Item Response Theory (IRT) and reinforcement learning to dynamically adjust evaluation strategies and improve accuracy over time.
        </Typography>
      </Alert>
    );
  }

  return (
    <Box>
      <Typography variant="h1" sx={{ 
        background: 'linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%)',
        WebkitBackgroundClip: 'text',
        WebkitTextFillColor: 'transparent',
        mb: 3,
      }}>
        Adaptive Analysis
      </Typography>

      <Box display="flex" flexDirection="column" gap={3}>
        {/* Overview Cards */}
        <Box display="flex" gap={2} flexWrap="wrap">
          <Card sx={{ 
            background: 'linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%)',
            color: 'white',
            textAlign: 'center',
            flex: '1',
            minWidth: '200px',
          }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Total Iterations
              </Typography>
              <Typography variant="h2" fontWeight="bold">
                {adaptiveData.total_iterations || 0}
              </Typography>
              <Typography variant="body2" sx={{ opacity: 0.9 }}>
                Adaptive cycles completed
              </Typography>
            </CardContent>
          </Card>

          <Card sx={{ 
            background: adaptiveData.convergence_achieved 
              ? 'linear-gradient(135deg, #4CAF50 0%, #45a049 100%)'
              : 'linear-gradient(135deg, #ff9800 0%, #f57c00 100%)',
            color: 'white',
            textAlign: 'center',
            flex: '1',
            minWidth: '200px',
          }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Convergence
              </Typography>
              <Typography variant="h3" fontWeight="bold">
                {adaptiveData.convergence_achieved ? '✓' : '○'}
              </Typography>
              <Typography variant="body2" sx={{ opacity: 0.9 }}>
                {adaptiveData.convergence_achieved ? 'Achieved' : 'In Progress'}
              </Typography>
            </CardContent>
          </Card>

          <Card sx={{ 
            background: 'linear-gradient(135deg, #A8E6CF 0%, #FF8B94 100%)',
            color: 'white',
            textAlign: 'center',
            flex: '1',
            minWidth: '200px',
          }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Final Accuracy
              </Typography>
              <Typography variant="h2" fontWeight="bold">
                {adaptiveData.final_accuracy ? `${(adaptiveData.final_accuracy * 100).toFixed(1)}%` : 'N/A'}
              </Typography>
              <Typography variant="body2" sx={{ opacity: 0.9 }}>
                Adaptive performance
              </Typography>
            </CardContent>
          </Card>
        </Box>

        {/* Available Reports */}
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Available Adaptive Reports
            </Typography>
            <Box display="flex" flexWrap="wrap" gap={1}>
              {adaptiveData.available_reports.map((report) => (
                <Chip 
                  key={report} 
                  label={report.replace(/_/g, ' ')} 
                  size="small" 
                  color="secondary"
                  variant="outlined"
                />
              ))}
            </Box>
          </CardContent>
        </Card>

        {/* Adaptive Features Info */}
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Adaptive Evaluation Features
            </Typography>
            <Box display="flex" flexDirection="column" gap={2}>
              <Alert severity="info">
                <Typography variant="body2">
                  <strong>Dynamic Difficulty:</strong> Tasks are adaptively selected based on agent ability estimates.
                </Typography>
              </Alert>
              <Alert severity="success">
                <Typography variant="body2">
                  <strong>Prompt Evolution:</strong> Evaluation prompts improve through reinforcement learning.
                </Typography>
              </Alert>
              <Alert severity="warning">
                <Typography variant="body2">
                  <strong>IRT Modeling:</strong> Statistical models estimate task difficulty and agent ability.
                </Typography>
              </Alert>
            </Box>
          </CardContent>
        </Card>
      </Box>
    </Box>
  );
};

export default AdaptiveAnalysis;