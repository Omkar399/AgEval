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
  Accordion,
  AccordionSummary,
  AccordionDetails,
  TextField,
  InputAdornment,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Grid,
  LinearProgress,
  Tooltip,
} from '@mui/material';
import {
  ExpandMore,
  Search,
  Task,
  Assessment,
  Category,
  Info,
  Code,
  Psychology,
  Description,
  Speed,
} from '@mui/icons-material';
import { ApiService, TasksData } from '../services/api';

interface TaskDetail {
  id: string;
  prompt: string;
  tier: string;
  category: string;
  difficulty?: number;
  expected_output?: string;
  metrics_used?: string[];
}

interface TaskPerformance {
  overall_score: number;
  metrics: Record<string, number>;
  judge_scores?: Record<string, Record<string, number>>;
}

const TaskBrowser: React.FC = () => {
  const [tasksData, setTasksData] = useState<TasksData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTier, setSelectedTier] = useState<string>('all');
  const [selectedTask, setSelectedTask] = useState<TaskDetail | null>(null);
  const [taskDialogOpen, setTaskDialogOpen] = useState(false);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await ApiService.getTasksData();
      setTasksData(data);
    } catch (err) {
      setError('Failed to load task data');
      console.error('Task browser load error:', err);
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

  if (!tasksData?.tasks || tasksData.tasks.length === 0) {
    return (
      <Alert severity="info" sx={{ m: 2 }}>
        No task data available. Run an evaluation first to see tasks and their performance.
      </Alert>
    );
  }

  const filteredTasks = tasksData.tasks.filter(task => {
    const matchesSearch = task.prompt.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         task.category.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesTier = selectedTier === 'all' || task.tier === selectedTier;
    return matchesSearch && matchesTier;
  });

  const tiers = Array.from(new Set(tasksData.tasks.map(task => task.tier)));
  const categories = Array.from(new Set(tasksData.tasks.map(task => task.category)));

  const getTierIcon = (tier: string) => {
    switch (tier.toLowerCase()) {
      case 'atomic': return <Psychology />;
      case 'compositional': return <Code />;
      case 'end-to-end': return <Description />;
      default: return <Task />;
    }
  };

  const getTierColor = (tier: string) => {
    switch (tier.toLowerCase()) {
      case 'atomic': return '#4CAF50';
      case 'compositional': return '#FF9800';
      case 'end-to-end': return '#F44336';
      default: return '#9E9E9E';
    }
  };

  const getPerformanceColor = (score: number) => {
    if (score >= 0.8) return 'success';
    if (score >= 0.6) return 'warning';
    return 'error';
  };

  const handleTaskClick = (task: TaskDetail) => {
    setSelectedTask(task);
    setTaskDialogOpen(true);
  };

  const getTaskPerformance = (taskId: string): TaskPerformance | null => {
    return tasksData?.task_performance?.[taskId] || null;
  };

  return (
    <Box>
      <Typography variant="h1" sx={{ 
        background: 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
        WebkitBackgroundClip: 'text',
        WebkitTextFillColor: 'transparent',
        mb: 3,
      }}>
        Task Browser & Performance
      </Typography>

      {/* Controls Section */}
      <Box sx={{ mb: 3, display: 'flex', flexDirection: 'column', gap: 2 }}>
        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'center' }}>
          <TextField
            placeholder="Search tasks..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            size="small"
            sx={{ minWidth: '300px' }}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <Search />
                </InputAdornment>
              ),
            }}
          />
          
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            <Button
              variant={selectedTier === 'all' ? 'contained' : 'outlined'}
              onClick={() => setSelectedTier('all')}
              size="small"
            >
              All Tiers
            </Button>
            {tiers.map(tier => (
              <Button
                key={tier}
                variant={selectedTier === tier ? 'contained' : 'outlined'}
                onClick={() => setSelectedTier(tier)}
                size="small"
                sx={{ 
                  backgroundColor: selectedTier === tier ? getTierColor(tier) : undefined,
                  borderColor: getTierColor(tier),
                  color: selectedTier === tier ? 'white' : getTierColor(tier),
                  '&:hover': {
                    backgroundColor: getTierColor(tier),
                    color: 'white',
                  }
                }}
              >
                {tier}
              </Button>
            ))}
          </Box>
        </Box>

        {/* Summary Stats */}
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ 
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              color: 'white',
            }}>
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h4" fontWeight="bold">
                  {filteredTasks.length}
                </Typography>
                <Typography variant="body2">
                  {searchQuery || selectedTier !== 'all' ? 'Filtered' : 'Total'} Tasks
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ 
              background: 'linear-gradient(135deg, #4CAF50 0%, #45a049 100%)',
              color: 'white',
            }}>
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h4" fontWeight="bold">
                  {categories.length}
                </Typography>
                <Typography variant="body2">
                  Categories
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ 
              background: 'linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%)',
              color: 'white',
            }}>
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h4" fontWeight="bold">
                  {tiers.length}
                </Typography>
                <Typography variant="body2">
                  Difficulty Tiers
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ 
              background: 'linear-gradient(135deg, #A8E6CF 0%, #FF8B94 100%)',
              color: 'white',
            }}>
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h4" fontWeight="bold">
                  {tasksData.task_performance ? 
                    Object.keys(tasksData.task_performance).length : 0}
                </Typography>
                <Typography variant="body2">
                  With Performance Data
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>

      {/* Tasks List */}
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {filteredTasks.map((task, index) => {
          const performance = getTaskPerformance(task.id);
          return (
            <Accordion key={task.id}>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, width: '100%' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {getTierIcon(task.tier)}
                    <Typography variant="h6">
                      Task {index + 1}
                    </Typography>
                  </Box>
                  
                  <Chip
                    label={task.tier}
                    size="small"
                    sx={{ 
                      backgroundColor: getTierColor(task.tier),
                      color: 'white',
                    }}
                  />
                  
                  <Chip
                    label={task.category}
                    size="small"
                    variant="outlined"
                  />
                  
                  {performance && (
                    <Chip
                      label={`${(performance.overall_score * 100).toFixed(1)}%`}
                      size="small"
                      color={getPerformanceColor(performance.overall_score)}
                    />
                  )}
                  
                  <Box sx={{ flexGrow: 1 }} />
                  
                  <Button
                    size="small"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleTaskClick(task);
                    }}
                    startIcon={<Info />}
                  >
                    Details
                  </Button>
                </Box>
              </AccordionSummary>
              
              <AccordionDetails>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  {/* Task Prompt */}
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="subtitle1" gutterBottom>
                        Task Prompt
                      </Typography>
                      <Typography variant="body2" sx={{ 
                        fontFamily: 'monospace',
                        backgroundColor: '#f5f5f5',
                        padding: 2,
                        borderRadius: 1,
                        whiteSpace: 'pre-wrap',
                      }}>
                        {task.prompt}
                      </Typography>
                    </CardContent>
                  </Card>
                  
                  {/* Performance Data */}
                  {performance && (
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle1" gutterBottom>
                          Performance Metrics
                        </Typography>
                        <Grid container spacing={2}>
                          <Grid item xs={12} md={6}>
                            <Typography variant="body2" color="textSecondary" gutterBottom>
                              Overall Score
                            </Typography>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <LinearProgress
                                variant="determinate"
                                value={performance.overall_score * 100}
                                sx={{ flexGrow: 1, height: 8, borderRadius: 4 }}
                                color={getPerformanceColor(performance.overall_score)}
                              />
                              <Typography variant="body2" fontWeight="bold">
                                {(performance.overall_score * 100).toFixed(1)}%
                              </Typography>
                            </Box>
                          </Grid>
                          
                          <Grid item xs={12} md={6}>
                            <Typography variant="body2" color="textSecondary" gutterBottom>
                              Metric Breakdown
                            </Typography>
                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                              {Object.entries(performance.metrics).map(([metric, score]) => (
                                <Tooltip 
                                  key={metric} 
                                  title={`${metric}: ${(score * 100).toFixed(1)}%`}
                                >
                                  <Chip
                                    label={`${metric}: ${(score * 100).toFixed(1)}%`}
                                    size="small"
                                    color={getPerformanceColor(score)}
                                    variant="outlined"
                                  />
                                </Tooltip>
                              ))}
                            </Box>
                          </Grid>
                        </Grid>
                      </CardContent>
                    </Card>
                  )}
                </Box>
              </AccordionDetails>
            </Accordion>
          );
        })}
      </Box>

      {/* Task Detail Dialog */}
      <Dialog 
        open={taskDialogOpen} 
        onClose={() => setTaskDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            {selectedTask && getTierIcon(selectedTask.tier)}
            <Typography variant="h6">
              Task Details
            </Typography>
            {selectedTask && (
              <Chip
                label={selectedTask.tier}
                size="small"
                sx={{ 
                  backgroundColor: getTierColor(selectedTask.tier),
                  color: 'white',
                }}
              />
            )}
          </Box>
        </DialogTitle>
        <DialogContent>
          {selectedTask && (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Typography variant="body1">
                <strong>Category:</strong> {selectedTask.category}
              </Typography>
              
              <Typography variant="body1">
                <strong>Prompt:</strong>
              </Typography>
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Typography variant="body2" sx={{ 
                  fontFamily: 'monospace',
                  whiteSpace: 'pre-wrap',
                }}>
                  {selectedTask.prompt}
                </Typography>
              </Paper>
              
              {getTaskPerformance(selectedTask.id) && (
                <Box>
                  <Typography variant="body1" gutterBottom>
                    <strong>Performance Analysis:</strong>
                  </Typography>
                  <TableContainer component={Paper} variant="outlined">
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Metric</TableCell>
                          <TableCell align="right">Score</TableCell>
                          <TableCell align="right">Performance</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {Object.entries(getTaskPerformance(selectedTask.id)!.metrics).map(([metric, score]) => (
                          <TableRow key={metric}>
                            <TableCell>{metric}</TableCell>
                            <TableCell align="right">{(score * 100).toFixed(1)}%</TableCell>
                            <TableCell align="right">
                              <Chip
                                label={score >= 0.8 ? 'Excellent' : score >= 0.6 ? 'Good' : 'Needs Improvement'}
                                size="small"
                                color={getPerformanceColor(score)}
                              />
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Box>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTaskDialogOpen(false)}>
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default TaskBrowser; 