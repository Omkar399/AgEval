import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Alert,
  CircularProgress,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
} from 'recharts';
import { ApiService, CalibrationData, CalibrationChartData } from '../services/api';

const Calibration: React.FC = () => {
  const [calibrationData, setCalibrationData] = useState<CalibrationData | null>(null);
  const [chartData, setChartData] = useState<CalibrationChartData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const [calibration, charts] = await Promise.all([
        ApiService.getCalibrationData().catch(() => null),
        ApiService.getCalibrationChartData().catch(() => null),
      ]);
      
      setCalibrationData(calibration);
      setChartData(charts);
    } catch (err) {
      setError('Failed to load calibration data');
      console.error('Calibration load error:', err);
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

  if (!calibrationData && !chartData) {
    return (
      <Alert severity="info" sx={{ m: 2 }}>
        No calibration data available. Run an evaluation with anchor tasks first.
      </Alert>
    );
  }

  // Prepare bias offset chart data
  const biasChartData = chartData?.bias_offsets 
    ? Object.entries(chartData.bias_offsets).flatMap(([judge, metrics]) =>
        Object.entries(metrics).map(([metric, offset]) => ({
          judge,
          metric,
          offset: offset as number,
        }))
      )
    : [];

  // Prepare sample reliability and confidence data (replace with actual data when available)
  const reliabilityData = [
    { metric: 'Accuracy', reliability: 0.85 },
    { metric: 'Precision', reliability: 0.78 },
    { metric: 'Recall', reliability: 0.82 },
  ];

  const confidenceData = [
    { range: '0-20%', count: 5 },
    { range: '20-40%', count: 8 },
    { range: '40-60%', count: 12 },
    { range: '60-80%', count: 15 },
    { range: '80-100%', count: 10 },
  ];

  return (
    <Box>
      <Typography variant="h1" sx={{ 
        background: 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
        WebkitBackgroundClip: 'text',
        WebkitTextFillColor: 'transparent',
        mb: 3,
      }}>
        Calibration & Reliability Analysis
      </Typography>

      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        {/* Bias Offsets Overview */}
        {chartData?.bias_offsets && (
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Judge Bias Offsets
              </Typography>
              <Typography variant="body2" color="textSecondary" paragraph>
                Systematic biases detected in judge scoring. Values show how much each judge typically scores above (+) or below (-) the gold standard.
              </Typography>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={biasChartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="metric" />
                  <YAxis 
                    tickFormatter={(value) => value.toFixed(2)}
                    domain={[-0.2, 0.2]}
                  />
                  <Tooltip 
                    formatter={(value: any) => [value.toFixed(3), 'Bias Offset']}
                  />
                  <Legend />
                  <Bar dataKey="offset" fill="#ff6b6b" name="Bias Offset" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}

        {/* Analysis Row */}
        <Box sx={{ 
          display: 'grid', 
          gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, 
          gap: 3 
        }}>
          {/* Bias Offsets Table */}
          {chartData?.bias_offsets && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Detailed Bias Offsets
                </Typography>
                <TableContainer component={Paper} variant="outlined">
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Judge</TableCell>
                        {chartData.bias_offsets && Object.keys(Object.values(chartData.bias_offsets)[0] || {}).map(metric => (
                          <TableCell key={metric} align="right">{metric}</TableCell>
                        ))}
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {chartData.bias_offsets && Object.entries(chartData.bias_offsets).map(([judge, metrics]) => (
                        <TableRow key={judge}>
                          <TableCell>{judge}</TableCell>
                          {Object.entries(metrics).map(([metric, offset]) => (
                            <TableCell 
                              key={metric} 
                              align="right"
                              sx={{
                                color: Math.abs(offset as number) > 0.1 ? 'error.main' : 'text.primary',
                                fontWeight: Math.abs(offset as number) > 0.1 ? 'bold' : 'normal',
                              }}
                            >
                              {(offset as number).toFixed(3)}
                            </TableCell>
                          ))}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          )}

          {/* Agreement Analysis */}
          {chartData?.agreement_analysis && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Inter-Judge Agreement
                </Typography>
                <Typography variant="body2" color="textSecondary" paragraph>
                  Analysis of agreement between judges on scoring.
                </Typography>
                <Box>
                  {Object.entries(chartData.agreement_analysis).map(([key, value]) => (
                    <Box key={key} display="flex" justifyContent="space-between" mb={1}>
                      <Typography variant="body2">{key.replace(/_/g, ' ')}</Typography>
                      <Typography variant="body2" fontWeight="bold">
                        {typeof value === 'number' ? value.toFixed(3) : JSON.stringify(value)}
                      </Typography>
                    </Box>
                  ))}
                </Box>
              </CardContent>
            </Card>
          )}
        </Box>

        {/* Reliability Metrics */}
        {chartData?.reliability_metrics && (
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Reliability Metrics
              </Typography>
              <Box sx={{ 
                display: 'grid', 
                gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr', md: 'repeat(3, 1fr)' }, 
                gap: 2 
              }}>
                {Object.entries(chartData.reliability_metrics).map(([metric, data]) => (
                  <Box key={metric} p={2} border={1} borderColor="grey.300" borderRadius={1}>
                    <Typography variant="subtitle2" gutterBottom>
                      {metric.replace(/_/g, ' ')}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      {typeof data === 'object' ? JSON.stringify(data, null, 2) : data}
                    </Typography>
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>
        )}

        {/* Calibration Insights */}
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Calibration Insights
            </Typography>
            <Box>
              <Alert severity="info" sx={{ mb: 2 }}>
                <Typography variant="body2">
                  <strong>Bias Calibration:</strong> The system detects and corrects systematic biases in judge scoring using anchor tasks with known gold standard scores.
                </Typography>
              </Alert>
              <Alert severity="warning" sx={{ mb: 2 }}>
                <Typography variant="body2">
                  <strong>High Bias Warning:</strong> Judges with bias offsets greater than ±0.1 may need attention or different prompting strategies.
                </Typography>
              </Alert>
              <Alert severity="success">
                <Typography variant="body2">
                  <strong>Reliability:</strong> Higher inter-judge agreement indicates more reliable and consistent evaluation metrics.
                </Typography>
              </Alert>
            </Box>
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Calibration Overview
            </Typography>
            <Box sx={{ 
              display: 'grid', 
              gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, 
              gap: 2 
            }}>
              <Box>
                <Typography variant="subtitle1" gutterBottom>
                  Overall Reliability
                </Typography>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={reliabilityData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="metric" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="reliability" stroke="#667eea" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </Box>
              
              <Box>
                <Typography variant="subtitle1" gutterBottom>
                  Confidence Distribution
                </Typography>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={confidenceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="range" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill="#667eea" />
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            </Box>
          </CardContent>
        </Card>

        {/* Judge Performance */}
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Judge Performance Metrics
            </Typography>
            <Box textAlign="center" p={2} border={1} borderColor="grey.300" borderRadius={1}>
              <Typography variant="body2" color="textSecondary">
                Judge performance data will be displayed here when available.
              </Typography>
            </Box>
          </CardContent>
        </Card>
      </Box>
    </Box>
  );
};

export default Calibration;