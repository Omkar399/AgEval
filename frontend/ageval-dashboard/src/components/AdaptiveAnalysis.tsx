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
          No adaptive evaluation data found. Use the <strong>"Run Adaptive Evaluation"</strong> button in the Dashboard to run the enhanced adaptive evaluation pipeline.
        </Typography>
        <Typography variant="body2">
          The adaptive system uses Item Response Theory (IRT) and reinforcement learning to dynamically adjust evaluation strategies and improve accuracy over time.
        </Typography>
      </Alert>
    );
  }

  // Check if we have incomplete adaptive data (0 iterations means it didn't run properly)
  const hasIncompleteData = adaptiveData.total_iterations === 0;

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

      {hasIncompleteData && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          <Typography variant="body2" paragraph>
            <strong>Adaptive Data Found But Incomplete:</strong> The adaptive evaluation data exists but shows 0 iterations, 
            indicating that the adaptive evaluation pipeline may not have completed successfully.
          </Typography>
          <Typography variant="body2">
            Try running a fresh adaptive evaluation using the <strong>"Run Adaptive Evaluation"</strong> button in the Dashboard to generate complete adaptive analysis data.
          </Typography>
        </Alert>
      )}

      <Box display="flex" flexDirection="column" gap={3}>
        {/* Overview Cards - Using dashboard.py style with Box layout */}
        <Box 
          display="flex" 
          flexWrap="wrap" 
          gap={2}
          sx={{
            '& > *': {
              flex: '1 1 300px',
              minWidth: '300px'
            }
          }}
        >
          <Card sx={{ 
            background: 'linear-gradient(135deg, #9C27B0 0%, #673AB7 100%)',
            color: 'white',
            textAlign: 'center',
            boxShadow: '0 4px 8px rgba(156, 39, 176, 0.3)',
            borderRadius: '10px'
          }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                🔄 Total Iterations
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
            boxShadow: adaptiveData.convergence_achieved 
              ? '0 4px 8px rgba(76, 175, 80, 0.3)'
              : '0 4px 8px rgba(255, 152, 0, 0.3)',
            borderRadius: '10px'
          }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                🎯 Convergence
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
            background: 'linear-gradient(135deg, #2196F3 0%, #1976D2 100%)',
            color: 'white',
            textAlign: 'center',
            boxShadow: '0 4px 8px rgba(33, 150, 243, 0.3)',
            borderRadius: '10px'
          }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                📊 Final Accuracy
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
        <Card sx={{ 
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          color: 'white',
          borderRadius: '10px'
        }}>
          <CardContent>
            <Typography variant="h6" gutterBottom sx={{ color: 'white' }}>
              📋 Available Adaptive Reports
            </Typography>
            <Box display="flex" flexWrap="wrap" gap={1}>
              {adaptiveData.available_reports.map((report) => (
                <Chip 
                  key={report} 
                  label={report.replace(/_/g, ' ')} 
                  size="small" 
                  sx={{
                    backgroundColor: 'rgba(255, 255, 255, 0.2)',
                    color: 'white',
                    border: '1px solid rgba(255, 255, 255, 0.3)',
                    '&:hover': {
                      backgroundColor: 'rgba(255, 255, 255, 0.3)',
                    }
                  }}
                />
              ))}
            </Box>
          </CardContent>
        </Card>

        {/* Adaptive Features Info */}
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              🧠 Adaptive Evaluation Features
            </Typography>
            <Box display="flex" flexDirection="column" gap={2}>
              <Alert severity="info" sx={{
                background: 'linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%)',
                border: 'none',
                borderLeft: '4px solid #2196F3'
              }}>
                <Typography variant="body2">
                  <strong>🎯 Dynamic Difficulty:</strong> Tasks are adaptively selected based on agent ability estimates using Item Response Theory (IRT).
                </Typography>
              </Alert>
              <Alert severity="success" sx={{
                background: 'linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%)',
                border: 'none',
                borderLeft: '4px solid #4CAF50'
              }}>
                <Typography variant="body2">
                  <strong>🔄 Prompt Evolution:</strong> Evaluation prompts improve through reinforcement learning and genetic algorithms.
                </Typography>
              </Alert>
              <Alert severity="warning" sx={{
                background: 'linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%)',
                border: 'none',
                borderLeft: '4px solid #ff9800'
              }}>
                <Typography variant="body2">
                  <strong>📈 IRT Modeling:</strong> Statistical models estimate task difficulty and agent ability for optimal challenge calibration.
                </Typography>
              </Alert>
              {hasIncompleteData && (
                <Alert severity="error" sx={{
                  background: 'linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%)',
                  border: 'none',
                  borderLeft: '4px solid #f44336'
                }}>
                  <Typography variant="body2">
                    <strong>⚠️ Data Status:</strong> Adaptive evaluation data is incomplete. The system needs to run a full adaptive evaluation cycle to generate meaningful insights.
                  </Typography>
                </Alert>
              )}
            </Box>
          </CardContent>
        </Card>
      </Box>
    </Box>
  );
};

export default AdaptiveAnalysis;