import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Alert,
  CircularProgress,
  Button,
  Chip,
  Paper,
  Divider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent,
  Switch,
  FormControlLabel,
  IconButton,
  Tooltip,
  Badge,
} from '@mui/material';
import {
  Terminal,
  PlayArrow,
  Pause,
  Clear,
  Download,
  FilterList,
  Refresh,
  Error,
  Warning,
  Info,
  CheckCircle,
  Settings,
  Visibility,
  VisibilityOff,
} from '@mui/icons-material';

interface LogEntry {
  timestamp: string;
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR' | 'SUCCESS';
  message: string;
  component?: string;
  evaluation_id?: string;
  phase?: string;
  judge?: string;
  task?: string;
}

interface EvaluationProgress {
  phase: string;
  progress: number;
  current_task?: string;
  current_judge?: string;
  total_tasks?: number;
  completed_tasks?: number;
  estimated_remaining?: number;
}

const RealTimeLogs: React.FC = () => {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const [filterLevel, setFilterLevel] = useState<string>('ALL');
  const [filterComponent, setFilterComponent] = useState<string>('ALL');
  const [progress, setProgress] = useState<EvaluationProgress | null>(null);
  const [isEvaluationRunning, setIsEvaluationRunning] = useState(false);
  
  const logsEndRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Mock log entries for demonstration
  const mockLogs: LogEntry[] = [
    {
      timestamp: new Date().toISOString(),
      level: 'INFO',
      message: '🚀 Starting AgEval pipeline evaluation...',
      component: 'Pipeline',
      phase: 'Initialization'
    },
    {
      timestamp: new Date(Date.now() - 1000).toISOString(),
      level: 'SUCCESS',
      message: '✅ Successfully loaded 9 evaluation tasks',
      component: 'TaskLoader',
      phase: 'Task Loading'
    },
    {
      timestamp: new Date(Date.now() - 2000).toISOString(),
      level: 'INFO',
      message: '⚖️ Initializing 3 judges: GPT-4o-mini, Claude-3.5-Sonnet, Gemini-1.5-Flash',
      component: 'JudgeManager',
      phase: 'Judge Setup'
    },
    {
      timestamp: new Date(Date.now() - 3000).toISOString(),
      level: 'WARNING',
      message: '⚠️ Judge GPT-4o-mini rate limit detected, implementing backoff...',
      component: 'Judge',
      judge: 'GPT-4o-mini'
    },
    {
      timestamp: new Date(Date.now() - 4000).toISOString(),
      level: 'DEBUG',
      message: 'Processing task "arithmetic_calculation" with atomic tier difficulty',
      component: 'TaskProcessor',
      task: 'arithmetic_calculation'
    },
  ];

  useEffect(() => {
    // Initialize with mock data
    setLogs(mockLogs);

    // Simulate WebSocket connection
    const connectWebSocket = () => {
      try {
        // In real implementation, connect to ws://localhost:8001/ws
        setIsConnected(true);
        
        // Simulate periodic log updates
        const interval = setInterval(() => {
          if (!isPaused) {
            addMockLogEntry();
          }
        }, 2000);

        return () => clearInterval(interval);
      } catch (error) {
        console.error('WebSocket connection failed:', error);
        setIsConnected(false);
      }
    };

    connectWebSocket();
  }, [isPaused]);

  const addMockLogEntry = () => {
    const mockMessages = [
      { level: 'INFO' as const, message: '📝 Generating response for task "code_generation"', component: 'Agent' },
      { level: 'SUCCESS' as const, message: '✅ Task completed successfully', component: 'TaskProcessor' },
      { level: 'INFO' as const, message: '🔍 Judge Claude-3.5-Sonnet evaluating response...', component: 'Judge', judge: 'Claude-3.5-Sonnet' },
      { level: 'DEBUG' as const, message: 'API response received in 1.2s', component: 'APIClient' },
      { level: 'WARNING' as const, message: '⚠️ High confidence variance detected in scoring', component: 'Calibration' },
      { level: 'INFO' as const, message: '📊 Computing inter-judge agreement metrics', component: 'Aggregation' },
      { level: 'SUCCESS' as const, message: '🎯 Phase 5 (Agent Output Generation) completed', component: 'Pipeline' },
    ];

    const randomMessage = mockMessages[Math.floor(Math.random() * mockMessages.length)];
    const newEntry: LogEntry = {
      timestamp: new Date().toISOString(),
      ...randomMessage,
    };

    setLogs(prev => [...prev, newEntry].slice(-100)); // Keep last 100 entries
  };

  useEffect(() => {
    if (autoScroll && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, autoScroll]);

  const handleFilterLevelChange = (event: SelectChangeEvent) => {
    setFilterLevel(event.target.value);
  };

  const handleFilterComponentChange = (event: SelectChangeEvent) => {
    setFilterComponent(event.target.value);
  };

  const clearLogs = () => {
    setLogs([]);
  };

  const downloadLogs = () => {
    const logText = filteredLogs.map(log => 
      `[${log.timestamp}] ${log.level} - ${log.component || 'System'}: ${log.message}`
    ).join('\n');
    
    const blob = new Blob([logText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ageval_logs_${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const getLevelIcon = (level: string) => {
    switch (level) {
      case 'ERROR': return <Error sx={{ color: '#f44336' }} />;
      case 'WARNING': return <Warning sx={{ color: '#ff9800' }} />;
      case 'SUCCESS': return <CheckCircle sx={{ color: '#4caf50' }} />;
      case 'INFO': return <Info sx={{ color: '#2196f3' }} />;
      case 'DEBUG': return <Settings sx={{ color: '#9e9e9e' }} />;
      default: return <Info sx={{ color: '#2196f3' }} />;
    }
  };

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'ERROR': return '#ffebee';
      case 'WARNING': return '#fff3e0';
      case 'SUCCESS': return '#e8f5e8';
      case 'INFO': return '#e3f2fd';
      case 'DEBUG': return '#f5f5f5';
      default: return '#ffffff';
    }
  };

  const getComponentColor = (component?: string) => {
    if (!component) return '#9e9e9e';
    const colors = {
      'Pipeline': '#9c27b0',
      'Judge': '#3f51b5',
      'Agent': '#4caf50',
      'TaskProcessor': '#ff9800',
      'Calibration': '#e91e63',
      'Aggregation': '#00bcd4',
    };
    return colors[component as keyof typeof colors] || '#9e9e9e';
  };

  const filteredLogs = logs.filter(log => {
    const levelMatch = filterLevel === 'ALL' || log.level === filterLevel;
    const componentMatch = filterComponent === 'ALL' || log.component === filterComponent;
    return levelMatch && componentMatch;
  });

  const components = Array.from(new Set(logs.map(log => log.component).filter(Boolean)));

  return (
    <Box>
      <Typography variant="h1" sx={{ 
        background: 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
        WebkitBackgroundClip: 'text',
        WebkitTextFillColor: 'transparent',
        mb: 3,
      }}>
        Real-Time Evaluation Logs
      </Typography>

      {/* Status and Controls */}
      <Box sx={{ mb: 3, display: 'flex', flexDirection: 'column', gap: 2 }}>
        {/* Connection Status */}
        <Card>
          <CardContent sx={{ py: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Badge color={isConnected ? 'success' : 'error'} variant="dot">
                  <Terminal />
                </Badge>
                <Typography variant="h6">
                  WebSocket Status: {isConnected ? 'Connected' : 'Disconnected'}
                </Typography>
                {isEvaluationRunning && (
                  <Chip 
                    label="Evaluation Running" 
                    color="primary"
                    icon={<CircularProgress size={16} color="inherit" />}
                  />
                )}
              </Box>
              
              <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={autoScroll}
                      onChange={(e) => setAutoScroll(e.target.checked)}
                      size="small"
                    />
                  }
                  label="Auto-scroll"
                />
                
                <Tooltip title={isPaused ? 'Resume' : 'Pause'}>
                  <IconButton 
                    onClick={() => setIsPaused(!isPaused)}
                    color={isPaused ? 'warning' : 'primary'}
                  >
                    {isPaused ? <PlayArrow /> : <Pause />}
                  </IconButton>
                </Tooltip>
                
                <Tooltip title="Clear logs">
                  <IconButton onClick={clearLogs} color="error">
                    <Clear />
                  </IconButton>
                </Tooltip>
                
                <Tooltip title="Download logs">
                  <IconButton onClick={downloadLogs} color="success">
                    <Download />
                  </IconButton>
                </Tooltip>
              </Box>
            </Box>
          </CardContent>
        </Card>

        {/* Filters */}
        <Card>
          <CardContent sx={{ py: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
              <FilterList />
              <Typography variant="body2" color="textSecondary">
                Filters:
              </Typography>
              
              <FormControl size="small" sx={{ minWidth: 120 }}>
                <InputLabel>Level</InputLabel>
                <Select value={filterLevel} onChange={handleFilterLevelChange} label="Level">
                  <MenuItem value="ALL">All Levels</MenuItem>
                  <MenuItem value="ERROR">Error</MenuItem>
                  <MenuItem value="WARNING">Warning</MenuItem>
                  <MenuItem value="SUCCESS">Success</MenuItem>
                  <MenuItem value="INFO">Info</MenuItem>
                  <MenuItem value="DEBUG">Debug</MenuItem>
                </Select>
              </FormControl>
              
              <FormControl size="small" sx={{ minWidth: 140 }}>
                <InputLabel>Component</InputLabel>
                <Select value={filterComponent} onChange={handleFilterComponentChange} label="Component">
                  <MenuItem value="ALL">All Components</MenuItem>
                  {components.map(component => (
                    <MenuItem key={component} value={component}>{component}</MenuItem>
                  ))}
                </Select>
              </FormControl>
              
              <Typography variant="body2" color="textSecondary">
                Showing {filteredLogs.length} of {logs.length} entries
              </Typography>
            </Box>
          </CardContent>
        </Card>

        {/* Progress Bar */}
        {progress && (
          <Card>
            <CardContent sx={{ py: 2 }}>
              <Box sx={{ mb: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="body2" fontWeight="bold">
                  {progress.phase}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  {progress.progress.toFixed(1)}%
                </Typography>
              </Box>
              <Box sx={{ width: '100%', mb: 1 }}>
                <Box
                  sx={{
                    width: '100%',
                    height: 8,
                    backgroundColor: '#e0e0e0',
                    borderRadius: 4,
                    overflow: 'hidden',
                  }}
                >
                  <Box
                    sx={{
                      width: `${progress.progress}%`,
                      height: '100%',
                      backgroundColor: '#4caf50',
                      transition: 'width 0.3s ease',
                    }}
                  />
                </Box>
              </Box>
              {(progress.current_task || progress.current_judge) && (
                <Typography variant="caption" color="textSecondary">
                  {progress.current_task && `Task: ${progress.current_task} `}
                  {progress.current_judge && `| Judge: ${progress.current_judge} `}
                  {progress.estimated_remaining && `| ETA: ${Math.round(progress.estimated_remaining / 60)}min`}
                </Typography>
              )}
            </CardContent>
          </Card>
        )}
      </Box>

      {/* Logs Display */}
      <Paper 
        sx={{ 
          height: '600px', 
          overflow: 'auto',
          backgroundColor: '#1a1a1a',
          fontFamily: 'Monaco, Menlo, "Ubuntu Mono", monospace',
          fontSize: '13px',
          border: '1px solid #333',
        }}
      >
        <Box sx={{ p: 2 }}>
          {filteredLogs.length === 0 ? (
            <Box sx={{ 
              display: 'flex', 
              justifyContent: 'center', 
              alignItems: 'center', 
              height: '200px',
              color: '#666'
            }}>
              <Typography variant="body2">
                {logs.length === 0 ? 'No logs available' : 'No logs match current filters'}
              </Typography>
            </Box>
          ) : (
            filteredLogs.map((log, index) => (
              <Box
                key={index}
                sx={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: 1,
                  mb: 1,
                  p: 1,
                  borderRadius: 1,
                  backgroundColor: getLevelColor(log.level),
                  border: `1px solid ${getLevelColor(log.level)}`,
                  '&:hover': {
                    backgroundColor: '#f0f0f0',
                  }
                }}
              >
                <Box sx={{ mt: 0.5 }}>
                  {getLevelIcon(log.level)}
                </Box>
                
                <Box sx={{ flexGrow: 1, minWidth: 0 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5, flexWrap: 'wrap' }}>
                    <Typography 
                      variant="caption" 
                      sx={{ 
                        fontFamily: 'monospace',
                        color: '#666',
                        fontSize: '11px'
                      }}
                    >
                      {new Date(log.timestamp).toLocaleTimeString()}
                    </Typography>
                    
                    <Chip
                      label={log.level}
                      size="small"
                      sx={{
                        height: '20px',
                        fontSize: '10px',
                        fontWeight: 'bold',
                        backgroundColor: getLevelIcon(log.level).props.sx?.color,
                        color: 'white',
                      }}
                    />
                    
                    {log.component && (
                      <Chip
                        label={log.component}
                        size="small"
                        variant="outlined"
                        sx={{
                          height: '20px',
                          fontSize: '10px',
                          borderColor: getComponentColor(log.component),
                          color: getComponentColor(log.component),
                        }}
                      />
                    )}
                    
                    {log.judge && (
                      <Chip
                        label={`Judge: ${log.judge}`}
                        size="small"
                        variant="outlined"
                        sx={{ height: '20px', fontSize: '10px' }}
                      />
                    )}
                    
                    {log.task && (
                      <Chip
                        label={`Task: ${log.task}`}
                        size="small"
                        variant="outlined"
                        sx={{ height: '20px', fontSize: '10px' }}
                      />
                    )}
                  </Box>
                  
                  <Typography 
                    variant="body2" 
                    sx={{ 
                      fontFamily: 'monospace',
                      color: '#333',
                      fontSize: '13px',
                      wordBreak: 'break-word',
                      whiteSpace: 'pre-wrap'
                    }}
                  >
                    {log.message}
                  </Typography>
                </Box>
              </Box>
            ))
          )}
          <div ref={logsEndRef} />
        </Box>
      </Paper>

      {isPaused && (
        <Alert severity="warning" sx={{ mt: 2 }}>
          Log streaming is paused. Click the play button to resume real-time updates.
        </Alert>
      )}
    </Box>
  );
};

export default RealTimeLogs; 