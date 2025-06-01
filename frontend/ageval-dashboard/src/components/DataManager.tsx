import React, { useState, useRef } from 'react';
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
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Tooltip,
  Grid,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  LinearProgress,
} from '@mui/material';
import {
  CloudDownload,
  CloudUpload,
  Backup,
  Restore,
  FileDownload,
  FileUpload,
  Delete,
  Folder,
  InsertDriveFile,
  ExpandMore,
  Settings,
  Assessment,
  DataObject,
  Schedule,
  Storage,
  Security,
  Refresh,
  CheckCircle,
  Error,
  Warning,
} from '@mui/icons-material';

interface ExportOptions {
  includeRawScores: boolean;
  includeCalibrationData: boolean;
  includeTaskDetails: boolean;
  includeAgentOutputs: boolean;
  includeJudgeAnalysis: boolean;
  format: 'json' | 'csv' | 'xlsx';
  compressionLevel: 'none' | 'standard' | 'maximum';
}

interface BackupItem {
  id: string;
  name: string;
  timestamp: string;
  size: string;
  type: 'full' | 'evaluation' | 'config';
  description: string;
  status: 'completed' | 'in_progress' | 'failed';
}

const DataManager: React.FC = () => {
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  const [importDialogOpen, setImportDialogOpen] = useState(false);
  const [backupDialogOpen, setBackupDialogOpen] = useState(false);
  const [exportOptions, setExportOptions] = useState<ExportOptions>({
    includeRawScores: true,
    includeCalibrationData: true,
    includeTaskDetails: true,
    includeAgentOutputs: false,
    includeJudgeAnalysis: true,
    format: 'json',
    compressionLevel: 'standard',
  });
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isExporting, setIsExporting] = useState(false);
  const [isImporting, setIsImporting] = useState(false);
  const [backupName, setBackupName] = useState('');
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Mock backup data
  const mockBackups: BackupItem[] = [
    {
      id: '1',
      name: 'Full Evaluation Backup - GPT-4 Analysis',
      timestamp: '2024-06-01T10:30:00Z',
      size: '15.2 MB',
      type: 'full',
      description: 'Complete evaluation data including all judges, tasks, and calibration',
      status: 'completed',
    },
    {
      id: '2',
      name: 'Configuration Backup - Production Settings',
      timestamp: '2024-06-01T08:15:00Z',
      size: '0.8 MB',
      type: 'config',
      description: 'Judge configurations, API keys, and evaluation parameters',
      status: 'completed',
    },
    {
      id: '3',
      name: 'Evaluation Results - Adaptive Analysis',
      timestamp: '2024-05-31T16:45:00Z',
      size: '8.7 MB',
      type: 'evaluation',
      description: 'Results from adaptive evaluation with IRT analysis',
      status: 'completed',
    },
  ];

  const handleExportOptionChange = (option: keyof ExportOptions, value: any) => {
    setExportOptions(prev => ({ ...prev, [option]: value }));
  };

  const handleExport = async () => {
    setIsExporting(true);
    try {
      // Simulate export process
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Generate mock data based on options
      const exportData = {
        metadata: {
          export_timestamp: new Date().toISOString(),
          ageval_version: '1.0.0',
          export_options: exportOptions,
        },
        ...(exportOptions.includeRawScores && { raw_scores: {} }),
        ...(exportOptions.includeCalibrationData && { calibration: {} }),
        ...(exportOptions.includeTaskDetails && { tasks: {} }),
        ...(exportOptions.includeAgentOutputs && { agent_outputs: {} }),
        ...(exportOptions.includeJudgeAnalysis && { judge_analysis: {} }),
      };

      // Create and download file
      const blob = new Blob([JSON.stringify(exportData, null, 2)], { 
        type: 'application/json' 
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `ageval_export_${new Date().toISOString().split('T')[0]}.${exportOptions.format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      setExportDialogOpen(false);
    } catch (error) {
      console.error('Export failed:', error);
    } finally {
      setIsExporting(false);
    }
  };

  const handleImport = async () => {
    if (!selectedFiles || selectedFiles.length === 0) return;
    
    setIsImporting(true);
    setUploadProgress(0);
    
    try {
      // Simulate upload progress
      for (let i = 0; i <= 100; i += 10) {
        setUploadProgress(i);
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      
      // Process imported data
      const file = selectedFiles[0];
      const text = await file.text();
      const data = JSON.parse(text);
      
      console.log('Imported data:', data);
      setImportDialogOpen(false);
      setSelectedFiles(null);
    } catch (error) {
      console.error('Import failed:', error);
    } finally {
      setIsImporting(false);
      setUploadProgress(0);
    }
  };

  const handleBackupCreate = async () => {
    if (!backupName.trim()) return;
    
    try {
      // Simulate backup creation
      await new Promise(resolve => setTimeout(resolve, 1500));
      console.log('Creating backup:', backupName);
      setBackupDialogOpen(false);
      setBackupName('');
    } catch (error) {
      console.error('Backup failed:', error);
    }
  };

  const handleBackupRestore = (backupId: string) => {
    console.log('Restoring backup:', backupId);
  };

  const handleBackupDelete = (backupId: string) => {
    console.log('Deleting backup:', backupId);
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSelectedFiles(event.target.files);
  };

  const getBackupIcon = (type: string) => {
    switch (type) {
      case 'full': return <Backup sx={{ color: '#4caf50' }} />;
      case 'evaluation': return <Assessment sx={{ color: '#2196f3' }} />;
      case 'config': return <Settings sx={{ color: '#ff9800' }} />;
      default: return <Folder />;
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle sx={{ color: '#4caf50' }} />;
      case 'in_progress': return <CircularProgress size={20} />;
      case 'failed': return <Error sx={{ color: '#f44336' }} />;
      default: return null;
    }
  };

  const formatDate = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  return (
    <Box>
      <Typography variant="h1" sx={{ 
        background: 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
        WebkitBackgroundClip: 'text',
        WebkitTextFillColor: 'transparent',
        mb: 3,
      }}>
        Data Management & Backup
      </Typography>

      {/* Quick Actions */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ 
            background: 'linear-gradient(135deg, #4CAF50 0%, #45a049 100%)',
            color: 'white',
            cursor: 'pointer',
            '&:hover': {
              transform: 'translateY(-2px)',
              boxShadow: '0 8px 16px rgba(76, 175, 80, 0.3)',
            },
            transition: 'all 0.3s ease',
          }}
          onClick={() => setExportDialogOpen(true)}
          >
            <CardContent sx={{ textAlign: 'center' }}>
              <CloudDownload sx={{ fontSize: 48, mb: 1 }} />
              <Typography variant="h6">Export Data</Typography>
              <Typography variant="body2" sx={{ opacity: 0.9 }}>
                Download evaluation results
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ 
            background: 'linear-gradient(135deg, #2196F3 0%, #1976D2 100%)',
            color: 'white',
            cursor: 'pointer',
            '&:hover': {
              transform: 'translateY(-2px)',
              boxShadow: '0 8px 16px rgba(33, 150, 243, 0.3)',
            },
            transition: 'all 0.3s ease',
          }}
          onClick={() => setImportDialogOpen(true)}
          >
            <CardContent sx={{ textAlign: 'center' }}>
              <CloudUpload sx={{ fontSize: 48, mb: 1 }} />
              <Typography variant="h6">Import Data</Typography>
              <Typography variant="body2" sx={{ opacity: 0.9 }}>
                Upload evaluation files
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ 
            background: 'linear-gradient(135deg, #FF9800 0%, #F57C00 100%)',
            color: 'white',
            cursor: 'pointer',
            '&:hover': {
              transform: 'translateY(-2px)',
              boxShadow: '0 8px 16px rgba(255, 152, 0, 0.3)',
            },
            transition: 'all 0.3s ease',
          }}
          onClick={() => setBackupDialogOpen(true)}
          >
            <CardContent sx={{ textAlign: 'center' }}>
              <Backup sx={{ fontSize: 48, mb: 1 }} />
              <Typography variant="h6">Create Backup</Typography>
              <Typography variant="body2" sx={{ opacity: 0.9 }}>
                Save current state
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ 
            background: 'linear-gradient(135deg, #9C27B0 0%, #7B1FA2 100%)',
            color: 'white',
            cursor: 'pointer',
            '&:hover': {
              transform: 'translateY(-2px)',
              boxShadow: '0 8px 16px rgba(156, 39, 176, 0.3)',
            },
            transition: 'all 0.3s ease',
          }}
          >
            <CardContent sx={{ textAlign: 'center' }}>
              <Storage sx={{ fontSize: 48, mb: 1 }} />
              <Typography variant="h6">Data Analytics</Typography>
              <Typography variant="body2" sx={{ opacity: 0.9 }}>
                Analyze usage patterns
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Backup Management */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
            <Typography variant="h6">
              Backup Management
            </Typography>
            <Button
              startIcon={<Refresh />}
              variant="outlined"
              size="small"
            >
              Refresh
            </Button>
          </Box>

          <List>
            {mockBackups.map((backup) => (
              <ListItem key={backup.id} divider>
                <ListItemIcon>
                  {getBackupIcon(backup.type)}
                </ListItemIcon>
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="subtitle1">
                        {backup.name}
                      </Typography>
                      <Chip 
                        label={backup.type}
                        size="small"
                        color={backup.type === 'full' ? 'primary' : backup.type === 'evaluation' ? 'secondary' : 'warning'}
                      />
                      {getStatusIcon(backup.status)}
                    </Box>
                  }
                  secondary={
                    <Box>
                      <Typography variant="body2" color="textSecondary">
                        {backup.description}
                      </Typography>
                      <Typography variant="caption" color="textSecondary">
                        Created: {formatDate(backup.timestamp)} • Size: {backup.size}
                      </Typography>
                    </Box>
                  }
                />
                <ListItemSecondaryAction>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <Tooltip title="Restore">
                      <IconButton 
                        onClick={() => handleBackupRestore(backup.id)}
                        color="primary"
                        size="small"
                      >
                        <Restore />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Download">
                      <IconButton 
                        color="success"
                        size="small"
                      >
                        <FileDownload />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Delete">
                      <IconButton 
                        onClick={() => handleBackupDelete(backup.id)}
                        color="error"
                        size="small"
                      >
                        <Delete />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </ListItemSecondaryAction>
              </ListItem>
            ))}
          </List>
        </CardContent>
      </Card>

      {/* Data Insights */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMore />}>
          <Typography variant="h6">Data Usage Insights</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="subtitle1" gutterBottom>
                  Storage Breakdown
                </Typography>
                <Box sx={{ mb: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">Evaluation Data</Typography>
                    <Typography variant="body2">12.4 MB</Typography>
                  </Box>
                  <LinearProgress variant="determinate" value={65} sx={{ mb: 2 }} />
                  
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">Cache Files</Typography>
                    <Typography variant="body2">8.1 MB</Typography>
                  </Box>
                  <LinearProgress variant="determinate" value={42} sx={{ mb: 2 }} />
                  
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">Configurations</Typography>
                    <Typography variant="body2">1.2 MB</Typography>
                  </Box>
                  <LinearProgress variant="determinate" value={6} />
                </Box>
              </Paper>
            </Grid>

            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="subtitle1" gutterBottom>
                  Recent Activity
                </Typography>
                <List dense>
                  <ListItem>
                    <ListItemIcon>
                      <Assessment fontSize="small" />
                    </ListItemIcon>
                    <ListItemText 
                      primary="Evaluation completed"
                      secondary="2 hours ago"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <Backup fontSize="small" />
                    </ListItemIcon>
                    <ListItemText 
                      primary="Backup created"
                      secondary="1 day ago"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <CloudDownload fontSize="small" />
                    </ListItemIcon>
                    <ListItemText 
                      primary="Data exported"
                      secondary="3 days ago"
                    />
                  </ListItem>
                </List>
              </Paper>
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Export Dialog */}
      <Dialog open={exportDialogOpen} onClose={() => setExportDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Export Evaluation Data</DialogTitle>
        <DialogContent>
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle1" gutterBottom>
              Export Options
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth size="small">
                  <InputLabel>Format</InputLabel>
                  <Select
                    value={exportOptions.format}
                    onChange={(e) => handleExportOptionChange('format', e.target.value)}
                    label="Format"
                  >
                    <MenuItem value="json">JSON</MenuItem>
                    <MenuItem value="csv">CSV</MenuItem>
                    <MenuItem value="xlsx">Excel (XLSX)</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth size="small">
                  <InputLabel>Compression</InputLabel>
                  <Select
                    value={exportOptions.compressionLevel}
                    onChange={(e) => handleExportOptionChange('compressionLevel', e.target.value)}
                    label="Compression"
                  >
                    <MenuItem value="none">None</MenuItem>
                    <MenuItem value="standard">Standard</MenuItem>
                    <MenuItem value="maximum">Maximum</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
          </Box>

          <Typography variant="subtitle1" gutterBottom>
            Include Data
          </Typography>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            {Object.entries(exportOptions).filter(([key]) => key.startsWith('include')).map(([key, value]) => (
              <Box key={key} sx={{ display: 'flex', alignItems: 'center' }}>
                <input
                  type="checkbox"
                  checked={value as boolean}
                  onChange={(e) => handleExportOptionChange(key as keyof ExportOptions, e.target.checked)}
                  style={{ marginRight: 8 }}
                />
                <Typography variant="body2">
                  {key.replace('include', '').replace(/([A-Z])/g, ' $1').trim()}
                </Typography>
              </Box>
            ))}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setExportDialogOpen(false)}>Cancel</Button>
          <Button 
            onClick={handleExport} 
            variant="contained"
            disabled={isExporting}
            startIcon={isExporting ? <CircularProgress size={20} /> : <FileDownload />}
          >
            {isExporting ? 'Exporting...' : 'Export'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Import Dialog */}
      <Dialog open={importDialogOpen} onClose={() => setImportDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Import Evaluation Data</DialogTitle>
        <DialogContent>
          <Box sx={{ mb: 3 }}>
            <Button
              variant="outlined"
              onClick={() => fileInputRef.current?.click()}
              startIcon={<FileUpload />}
              fullWidth
              sx={{ mb: 2 }}
            >
              Select Files
            </Button>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileSelect}
              accept=".json,.csv,.xlsx"
              multiple
              style={{ display: 'none' }}
            />
            
            {selectedFiles && selectedFiles.length > 0 && (
              <Paper sx={{ p: 2, backgroundColor: '#f5f5f5' }}>
                <Typography variant="subtitle2" gutterBottom>
                  Selected Files:
                </Typography>
                {Array.from(selectedFiles).map((file, index) => (
                  <Box key={index} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <InsertDriveFile fontSize="small" />
                    <Typography variant="body2">
                      {file.name} ({(file.size / 1024).toFixed(1)} KB)
                    </Typography>
                  </Box>
                ))}
              </Paper>
            )}
            
            {isImporting && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="body2" gutterBottom>
                  Upload Progress: {uploadProgress}%
                </Typography>
                <LinearProgress variant="determinate" value={uploadProgress} />
              </Box>
            )}
          </Box>

          <Alert severity="info">
            Importing data will merge with existing evaluation results. 
            Create a backup before proceeding if you want to preserve current data.
          </Alert>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setImportDialogOpen(false)}>Cancel</Button>
          <Button 
            onClick={handleImport} 
            variant="contained"
            disabled={!selectedFiles || selectedFiles.length === 0 || isImporting}
            startIcon={isImporting ? <CircularProgress size={20} /> : <FileUpload />}
          >
            {isImporting ? 'Importing...' : 'Import'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Backup Dialog */}
      <Dialog open={backupDialogOpen} onClose={() => setBackupDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Create New Backup</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            label="Backup Name"
            value={backupName}
            onChange={(e) => setBackupName(e.target.value)}
            placeholder="e.g., Pre-experiment backup"
            sx={{ mb: 2, mt: 1 }}
          />
          
          <Alert severity="info">
            This will create a complete backup of all evaluation data, configurations, and results.
          </Alert>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setBackupDialogOpen(false)}>Cancel</Button>
          <Button 
            onClick={handleBackupCreate} 
            variant="contained"
            disabled={!backupName.trim()}
            startIcon={<Backup />}
          >
            Create Backup
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default DataManager; 