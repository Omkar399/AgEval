import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Alert,
  CircularProgress,
  Chip,
  Grid,
} from '@mui/material';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import { ApiService, PerformanceData, RadarChartData, TasksData } from '../services/api';

const Performance: React.FC = () => {
  const [performanceData, setPerformanceData] = useState<PerformanceData | null>(null);
  const [radarData, setRadarData] = useState<RadarChartData | null>(null);
  const [tasksData, setTasksData] = useState<TasksData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const [performance, radar, tasks] = await Promise.all([
        ApiService.getPerformanceData().catch(() => null),
        ApiService.getPerformanceRadarData().catch(() => null),
        ApiService.getTasksData().catch(() => null),
      ]);
      
      setPerformanceData(performance);
      setRadarData(radar);
      setTasksData(tasks);
    } catch (err) {
      setError('Failed to load performance data');
      console.error('Performance load error:', err);
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

  if (!performanceData?.final_performance) {
    return (
      <Alert severity="info" sx={{ m: 2 }}>
        No performance data available. Run an evaluation first.
      </Alert>
    );
  }

  // Prepare radar chart data
  const radarChartData = radarData ? radarData.metrics.map((metric, index) => ({
    metric,
    score: radarData.scores[index],
    fullMark: 1.0,
  })) : [];

  // Prepare task performance chart data
  const taskChartData = tasksData?.task_performance 
    ? Object.entries(tasksData.task_performance).map(([taskId, data]) => ({
        task: taskId,
        score: data.overall_score,
        ...data.metrics,
      }))
    : [];

  // Calculate overall score
  const overallScore = performanceData.final_performance 
    ? Object.values(performanceData.final_performance).reduce((sum, score) => sum + score, 0) / Object.keys(performanceData.final_performance).length
    : 0;

  return (
    <Box>
      <Typography variant="h1" sx={{ 
        background: 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
        WebkitBackgroundClip: 'text',
        WebkitTextFillColor: 'transparent',
        mb: 3,
      }}>
        Performance Analysis
      </Typography>

      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        {/* Overall Score */}
        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 2fr' }, gap: 3 }}>
          <Card sx={{ 
            background: 'linear-gradient(135deg, #4CAF50 0%, #45a049 100%)',
            color: 'white',
            textAlign: 'center',
            py: 3,
          }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Overall Performance
              </Typography>
              <Typography variant="h2" fontWeight="bold">
                {(overallScore * 100).toFixed(1)}%
              </Typography>
              <Typography variant="body2" sx={{ opacity: 0.9 }}>
                Average across all metrics
              </Typography>
            </CardContent>
          </Card>

          {/* Metric Scores */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Metric Scores
              </Typography>
              <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 2 }}>
                {performanceData.final_performance && Object.entries(performanceData.final_performance).map(([metric, score]) => (
                  <Box key={metric} display="flex" justifyContent="space-between" alignItems="center" p={1}>
                    <Typography variant="body2">{metric}</Typography>
                    <Chip 
                      label={`${(score * 100).toFixed(1)}%`}
                      color={score > 0.8 ? 'success' : score > 0.6 ? 'warning' : 'error'}
                      size="small"
                    />
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Box>

        {/* Charts Row */}
        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 3 }}>
          {/* Radar Chart */}
          {radarChartData.length > 0 && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Performance Radar
                </Typography>
                <ResponsiveContainer width="100%" height={400}>
                  <RadarChart data={radarChartData}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="metric" tick={{ fontSize: 12 }} />
                    <PolarRadiusAxis 
                      angle={90} 
                      domain={[0, 1]} 
                      tick={{ fontSize: 10 }}
                      tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                    />
                    <Radar
                      name="Score"
                      dataKey="score"
                      stroke="#667eea"
                      fill="#667eea"
                      fillOpacity={0.3}
                      strokeWidth={2}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          {/* Task Performance */}
          {taskChartData.length > 0 && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Task Performance
                </Typography>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={taskChartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="task" 
                      angle={-45}
                      textAnchor="end"
                      height={100}
                      tick={{ fontSize: 10 }}
                    />
                    <YAxis 
                      domain={[0, 1]}
                      tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                    />
                    <Tooltip 
                      formatter={(value: any) => [`${(value * 100).toFixed(1)}%`, 'Score']}
                    />
                    <Legend />
                    <Bar dataKey="score" fill="#667eea" name="Overall Score" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}
        </Box>

        {/* Metrics Definitions */}
        {performanceData.canonical_metrics && (
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Metric Definitions
              </Typography>
              <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 2 }}>
                {performanceData.canonical_metrics.map((metric, index) => (
                  <Box key={index} p={2} border={1} borderColor="grey.300" borderRadius={1}>
                    <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                      {metric.name}
                    </Typography>
                    <Typography variant="body2" color="textSecondary" paragraph>
                      {metric.definition}
                    </Typography>
                    <Chip label={metric.scale} size="small" variant="outlined" />
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>
        )}

        {/* Performance Summary */}
        {performanceData.performance_summary && (
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Performance Summary
              </Typography>
              <Typography variant="body2" color="textSecondary">
                {JSON.stringify(performanceData.performance_summary, null, 2)}
              </Typography>
            </CardContent>
          </Card>
        )}
      </Box>
    </Box>
  );
};

export default Performance;