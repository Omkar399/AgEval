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
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent,
  Divider,
  LinearProgress,
  Tooltip,
} from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ScatterChart,
  Scatter,
  Cell,
  PieChart,
  Pie,
} from 'recharts';
import {
  Gavel,
  Psychology,
  Analytics,
  TrendingUp,
  CompareArrows,
  Assessment,
  Star,
  Warning,
} from '@mui/icons-material';
import { ApiService, CalibrationData } from '../services/api';

interface JudgeMetrics {
  name: string;
  bias_offset: number;
  reliability: number;
  agreement_score: number;
  scoring_variance: number;
  harsh_rating: number; // 0-1 scale where 1 is very harsh
  consistency: number;
}

interface JudgeComparison {
  judge1: string;
  judge2: string;
  agreement_rate: number;
  correlation: number;
  disagreement_patterns: string[];
}

const JudgeComparison: React.FC = () => {
  const [calibrationData, setCalibrationData] = useState<CalibrationData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedJudge1, setSelectedJudge1] = useState<string>('');
  const [selectedJudge2, setSelectedJudge2] = useState<string>('');
  const [viewMode, setViewMode] = useState<'overview' | 'detailed' | 'side-by-side'>('overview');

  // Mock data for demonstration - replace with real data when available
  const mockJudgeMetrics: JudgeMetrics[] = [
    {
      name: 'GPT-4o-mini',
      bias_offset: 0.02,
      reliability: 0.89,
      agreement_score: 0.76,
      scoring_variance: 0.15,
      harsh_rating: 0.3,
      consistency: 0.85,
    },
    {
      name: 'Claude-3.5-Sonnet',
      bias_offset: -0.05,
      reliability: 0.92,
      agreement_score: 0.81,
      scoring_variance: 0.12,
      harsh_rating: 0.7,
      consistency: 0.88,
    },
    {
      name: 'Gemini-1.5-Flash',
      bias_offset: 0.08,
      reliability: 0.84,
      agreement_score: 0.73,
      scoring_variance: 0.18,
      harsh_rating: 0.4,
      consistency: 0.82,
    },
  ];

  const mockComparisons: JudgeComparison[] = [
    {
      judge1: 'GPT-4o-mini',
      judge2: 'Claude-3.5-Sonnet',
      agreement_rate: 0.76,
      correlation: 0.82,
      disagreement_patterns: ['Technical precision', 'Response accuracy'],
    },
    {
      judge1: 'GPT-4o-mini',
      judge2: 'Gemini-1.5-Flash',
      agreement_rate: 0.71,
      correlation: 0.78,
      disagreement_patterns: ['Completeness', 'Reasoning clarity'],
    },
    {
      judge1: 'Claude-3.5-Sonnet',
      judge2: 'Gemini-1.5-Flash',
      agreement_rate: 0.69,
      correlation: 0.74,
      disagreement_patterns: ['Technical precision', 'Completeness'],
    },
  ];

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await ApiService.getCalibrationData();
      setCalibrationData(data);
      
      // Set default judges if available
      if (mockJudgeMetrics.length >= 2) {
        setSelectedJudge1(mockJudgeMetrics[0].name);
        setSelectedJudge2(mockJudgeMetrics[1].name);
      }
    } catch (err) {
      setError('Failed to load calibration data');
      console.error('Judge comparison load error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleJudge1Change = (event: SelectChangeEvent) => {
    setSelectedJudge1(event.target.value);
  };

  const handleJudge2Change = (event: SelectChangeEvent) => {
    setSelectedJudge2(event.target.value);
  };

  const getJudgeMetrics = (judgeName: string): JudgeMetrics | undefined => {
    return mockJudgeMetrics.find(j => j.name === judgeName);
  };

  const getJudgeComparison = (judge1: string, judge2: string): JudgeComparison | undefined => {
    return mockComparisons.find(c => 
      (c.judge1 === judge1 && c.judge2 === judge2) ||
      (c.judge1 === judge2 && c.judge2 === judge1)
    );
  };

  const getBiasColor = (bias: number) => {
    if (Math.abs(bias) < 0.03) return '#4CAF50'; // Green - minimal bias
    if (Math.abs(bias) < 0.06) return '#FF9800'; // Orange - moderate bias
    return '#F44336'; // Red - high bias
  };

  const getReliabilityColor = (reliability: number) => {
    if (reliability >= 0.85) return '#4CAF50'; // Green - high reliability
    if (reliability >= 0.75) return '#FF9800'; // Orange - moderate reliability
    return '#F44336'; // Red - low reliability
  };

  const formatBias = (bias: number) => {
    const sign = bias >= 0 ? '+' : '';
    return `${sign}${(bias * 100).toFixed(1)}%`;
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  const judgeNames = mockJudgeMetrics.map(j => j.name);

  // Prepare radar chart data
  const radarData = mockJudgeMetrics.map(judge => ({
    judge: judge.name.split('-')[0], // Short name
    reliability: judge.reliability * 100,
    agreement: judge.agreement_score * 100,
    consistency: judge.consistency * 100,
    bias: (1 - Math.abs(judge.bias_offset) * 10) * 100, // Inverted bias for radar
  }));

  // Prepare bias comparison data
  const biasData = mockJudgeMetrics.map(judge => ({
    name: judge.name.split('-')[0],
    bias: judge.bias_offset * 100,
    color: getBiasColor(judge.bias_offset),
  }));

  return (
    <Box>
      <Typography variant="h1" sx={{ 
        background: 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
        WebkitBackgroundClip: 'text',
        WebkitTextFillColor: 'transparent',
        mb: 3,
      }}>
        Judge Analysis & Comparison
      </Typography>

      {/* Controls */}
      <Box sx={{ mb: 3, display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'center' }}>
        <Button
          variant={viewMode === 'overview' ? 'contained' : 'outlined'}
          onClick={() => setViewMode('overview')}
          startIcon={<Assessment />}
        >
          Overview
        </Button>
        <Button
          variant={viewMode === 'detailed' ? 'contained' : 'outlined'}
          onClick={() => setViewMode('detailed')}
          startIcon={<Analytics />}
        >
          Detailed Analysis
        </Button>
        <Button
          variant={viewMode === 'side-by-side' ? 'contained' : 'outlined'}
          onClick={() => setViewMode('side-by-side')}
          startIcon={<CompareArrows />}
        >
          Side-by-Side
        </Button>

        {viewMode === 'side-by-side' && (
          <>
            <FormControl size="small" sx={{ minWidth: 150 }}>
              <InputLabel>Judge 1</InputLabel>
              <Select value={selectedJudge1} onChange={handleJudge1Change} label="Judge 1">
                {judgeNames.map(name => (
                  <MenuItem key={name} value={name}>{name}</MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl size="small" sx={{ minWidth: 150 }}>
              <InputLabel>Judge 2</InputLabel>
              <Select value={selectedJudge2} onChange={handleJudge2Change} label="Judge 2">
                {judgeNames.filter(name => name !== selectedJudge1).map(name => (
                  <MenuItem key={name} value={name}>{name}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </>
        )}
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Overview Mode */}
      {viewMode === 'overview' && (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          {/* Judge Summary Cards */}
          <Grid container spacing={3}>
            {mockJudgeMetrics.map((judge) => (
              <Grid item xs={12} md={4} key={judge.name}>
                <Card sx={{ height: '100%' }}>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <Gavel sx={{ mr: 1 }} />
                      <Typography variant="h6">{judge.name}</Typography>
                    </Box>
                    
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="body2">Reliability:</Typography>
                        <Chip
                          label={`${(judge.reliability * 100).toFixed(1)}%`}
                          size="small"
                          sx={{ backgroundColor: getReliabilityColor(judge.reliability), color: 'white' }}
                        />
                      </Box>
                      
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="body2">Bias Offset:</Typography>
                        <Chip
                          label={formatBias(judge.bias_offset)}
                          size="small"
                          sx={{ backgroundColor: getBiasColor(judge.bias_offset), color: 'white' }}
                        />
                      </Box>
                      
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="body2">Agreement:</Typography>
                        <Typography variant="body2" fontWeight="bold">
                          {(judge.agreement_score * 100).toFixed(1)}%
                        </Typography>
                      </Box>
                      
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="body2">Consistency:</Typography>
                        <Typography variant="body2" fontWeight="bold">
                          {(judge.consistency * 100).toFixed(1)}%
                        </Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>

          {/* Comparison Charts */}
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Judge Performance Radar
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <RadarChart data={radarData}>
                      <PolarGrid />
                      <PolarAngleAxis dataKey="judge" />
                      <PolarRadiusAxis angle={90} domain={[0, 100]} />
                      <Radar name="Reliability" dataKey="reliability" stroke="#4CAF50" fill="#4CAF50" fillOpacity={0.1} />
                      <Radar name="Agreement" dataKey="agreement" stroke="#FF9800" fill="#FF9800" fillOpacity={0.1} />
                      <Radar name="Consistency" dataKey="consistency" stroke="#2196F3" fill="#2196F3" fillOpacity={0.1} />
                      <Legend />
                    </RadarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Bias Comparison
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={biasData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis tickFormatter={(value) => `${value}%`} />
                      <RechartsTooltip formatter={(value: any) => [`${value.toFixed(1)}%`, 'Bias Offset']} />
                      <Bar dataKey="bias" fill="#8884d8">
                        {biasData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Agreement Matrix */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Inter-Judge Agreement Matrix
              </Typography>
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Judge Pair</TableCell>
                      <TableCell align="center">Agreement Rate</TableCell>
                      <TableCell align="center">Correlation</TableCell>
                      <TableCell>Main Disagreements</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {mockComparisons.map((comp, index) => (
                      <TableRow key={index}>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Typography variant="body2">
                              {comp.judge1.split('-')[0]} vs {comp.judge2.split('-')[0]}
                            </Typography>
                          </Box>
                        </TableCell>
                        <TableCell align="center">
                          <Chip
                            label={`${(comp.agreement_rate * 100).toFixed(1)}%`}
                            size="small"
                            color={comp.agreement_rate >= 0.75 ? 'success' : comp.agreement_rate >= 0.65 ? 'warning' : 'error'}
                          />
                        </TableCell>
                        <TableCell align="center">
                          <Typography variant="body2" fontWeight="bold">
                            {comp.correlation.toFixed(3)}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                            {comp.disagreement_patterns.map((pattern, i) => (
                              <Chip key={i} label={pattern} size="small" variant="outlined" />
                            ))}
                          </Box>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Box>
      )}

      {/* Side-by-Side Mode */}
      {viewMode === 'side-by-side' && selectedJudge1 && selectedJudge2 && (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          <Grid container spacing={3}>
            {/* Judge 1 */}
            <Grid item xs={12} md={6}>
              <Card sx={{ height: '100%', border: '2px solid #4CAF50' }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom sx={{ color: '#4CAF50' }}>
                    {selectedJudge1}
                  </Typography>
                  {(() => {
                    const metrics = getJudgeMetrics(selectedJudge1);
                    return metrics ? (
                      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                        <Box>
                          <Typography variant="body2" color="textSecondary" gutterBottom>
                            Reliability Score
                          </Typography>
                          <LinearProgress
                            variant="determinate"
                            value={metrics.reliability * 100}
                            sx={{ height: 8, borderRadius: 4 }}
                            color="success"
                          />
                          <Typography variant="body2" sx={{ mt: 1 }}>
                            {(metrics.reliability * 100).toFixed(1)}%
                          </Typography>
                        </Box>
                        
                        <Box>
                          <Typography variant="body2" color="textSecondary" gutterBottom>
                            Bias Offset
                          </Typography>
                          <Typography variant="h4" sx={{ color: getBiasColor(metrics.bias_offset) }}>
                            {formatBias(metrics.bias_offset)}
                          </Typography>
                        </Box>
                        
                        <Box>
                          <Typography variant="body2" color="textSecondary" gutterBottom>
                            Scoring Consistency
                          </Typography>
                          <LinearProgress
                            variant="determinate"
                            value={metrics.consistency * 100}
                            sx={{ height: 8, borderRadius: 4 }}
                            color="info"
                          />
                          <Typography variant="body2" sx={{ mt: 1 }}>
                            {(metrics.consistency * 100).toFixed(1)}%
                          </Typography>
                        </Box>
                      </Box>
                    ) : null;
                  })()}
                </CardContent>
              </Card>
            </Grid>

            {/* Judge 2 */}
            <Grid item xs={12} md={6}>
              <Card sx={{ height: '100%', border: '2px solid #FF9800' }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom sx={{ color: '#FF9800' }}>
                    {selectedJudge2}
                  </Typography>
                  {(() => {
                    const metrics = getJudgeMetrics(selectedJudge2);
                    return metrics ? (
                      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                        <Box>
                          <Typography variant="body2" color="textSecondary" gutterBottom>
                            Reliability Score
                          </Typography>
                          <LinearProgress
                            variant="determinate"
                            value={metrics.reliability * 100}
                            sx={{ height: 8, borderRadius: 4 }}
                            color="warning"
                          />
                          <Typography variant="body2" sx={{ mt: 1 }}>
                            {(metrics.reliability * 100).toFixed(1)}%
                          </Typography>
                        </Box>
                        
                        <Box>
                          <Typography variant="body2" color="textSecondary" gutterBottom>
                            Bias Offset
                          </Typography>
                          <Typography variant="h4" sx={{ color: getBiasColor(metrics.bias_offset) }}>
                            {formatBias(metrics.bias_offset)}
                          </Typography>
                        </Box>
                        
                        <Box>
                          <Typography variant="body2" color="textSecondary" gutterBottom>
                            Scoring Consistency
                          </Typography>
                          <LinearProgress
                            variant="determinate"
                            value={metrics.consistency * 100}
                            sx={{ height: 8, borderRadius: 4 }}
                            color="info"
                          />
                          <Typography variant="body2" sx={{ mt: 1 }}>
                            {(metrics.consistency * 100).toFixed(1)}%
                          </Typography>
                        </Box>
                      </Box>
                    ) : null;
                  })()}
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Direct Comparison */}
          {(() => {
            const comparison = getJudgeComparison(selectedJudge1, selectedJudge2);
            return comparison ? (
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Direct Comparison Analysis
                  </Typography>
                  <Grid container spacing={3}>
                    <Grid item xs={12} md={4}>
                      <Box sx={{ textAlign: 'center' }}>
                        <Typography variant="body2" color="textSecondary" gutterBottom>
                          Agreement Rate
                        </Typography>
                        <Typography variant="h3" sx={{ color: '#2196F3' }}>
                          {(comparison.agreement_rate * 100).toFixed(1)}%
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={12} md={4}>
                      <Box sx={{ textAlign: 'center' }}>
                        <Typography variant="body2" color="textSecondary" gutterBottom>
                          Correlation Coefficient
                        </Typography>
                        <Typography variant="h3" sx={{ color: '#9C27B0' }}>
                          {comparison.correlation.toFixed(3)}
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={12} md={4}>
                      <Box>
                        <Typography variant="body2" color="textSecondary" gutterBottom>
                          Main Disagreement Areas
                        </Typography>
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                          {comparison.disagreement_patterns.map((pattern, i) => (
                            <Chip 
                              key={i} 
                              label={pattern} 
                              size="small" 
                              color="warning"
                              icon={<Warning />}
                            />
                          ))}
                        </Box>
                      </Box>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            ) : null;
          })()}
        </Box>
      )}

      {/* Detailed Analysis Mode */}
      {viewMode === 'detailed' && (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          <Alert severity="info">
            <Typography variant="body2">
              Detailed analysis includes statistical significance tests, scoring pattern analysis, and bias correction recommendations.
              This view will show comprehensive judge behavior analysis when more evaluation data is available.
            </Typography>
          </Alert>
          
          {/* Placeholder for detailed analysis */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Statistical Analysis (Coming Soon)
              </Typography>
              <Typography variant="body2" color="textSecondary">
                This section will include advanced statistical analysis such as:
              </Typography>
              <Box component="ul" sx={{ mt: 2 }}>
                <Typography component="li" variant="body2">ANOVA tests for judge differences</Typography>
                <Typography component="li" variant="body2">Confidence interval analysis</Typography>
                <Typography component="li" variant="body2">Cronbach's alpha for internal consistency</Typography>
                <Typography component="li" variant="body2">Distribution analysis of scoring patterns</Typography>
                <Typography component="li" variant="body2">Bias correction recommendations</Typography>
              </Box>
            </CardContent>
          </Card>
        </Box>
      )}
    </Box>
  );
};

export default JudgeComparison; 