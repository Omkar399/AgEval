import React, { useState, useEffect } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Tabs,
  Tab,
  Box,
  CssBaseline,
  ThemeProvider,
  createTheme,
  Paper,
} from '@mui/material';
import { 
  Assessment, 
  Analytics, 
  Settings, 
  Timeline, 
  Task,
  Gavel,
  Terminal,
  Storage,
  Psychology,
} from '@mui/icons-material';
import Dashboard from './components/Dashboard';
import Performance from './components/Performance';
import Calibration from './components/Calibration';
import AdaptiveAnalysis from './components/AdaptiveAnalysis';
import TaskBrowser from './components/TaskBrowser';
import JudgeComparison from './components/JudgeComparison';
import RealTimeLogs from './components/RealTimeLogs';
import DataManager from './components/DataManager';
import AgentCards from './components/AgentCards';
import { ApiService } from './services/api';
import './App.css';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#667eea',
    },
    secondary: {
      main: '#764ba2',
    },
  },
  typography: {
    h1: {
      fontSize: '2.5rem',
      fontWeight: 'bold',
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 'bold',
    },
  },
});

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

function App() {
  const [tabValue, setTabValue] = useState(0);
  const [apiHealth, setApiHealth] = useState<boolean>(false);

  useEffect(() => {
    const checkApiHealth = async () => {
      try {
        await ApiService.healthCheck();
        setApiHealth(true);
      } catch (error) {
        setApiHealth(false);
        console.error('API health check failed:', error);
      }
    };

    checkApiHealth();
    const interval = setInterval(checkApiHealth, 30000); // Check every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <div className="App">
        <AppBar position="static" sx={{ background: 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)' }}>
          <Toolbar>
            <Assessment sx={{ mr: 2 }} />
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              AgEval Dashboard
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Box
                sx={{
                  width: 12,
                  height: 12,
                  borderRadius: '50%',
                  backgroundColor: apiHealth ? '#4caf50' : '#f44336',
                  mr: 1,
                }}
              />
              <Typography variant="body2">
                API {apiHealth ? 'Connected' : 'Disconnected'}
              </Typography>
            </Box>
          </Toolbar>
        </AppBar>

        <Container maxWidth="xl" sx={{ mt: 2 }}>
          <Paper elevation={1} sx={{ borderRadius: 2 }}>
            <Tabs 
              value={tabValue} 
              onChange={handleTabChange} 
              variant="scrollable"
              scrollButtons="auto"
              sx={{
                borderBottom: 1,
                borderColor: 'divider',
                '& .MuiTab-root': {
                  minHeight: 72,
                  textTransform: 'none',
                  fontSize: '0.9rem',
                  fontWeight: 'bold',
                  minWidth: 140,
                },
              }}
            >
              <Tab 
                icon={<Assessment />} 
                label="Dashboard" 
                iconPosition="start"
                sx={{ gap: 1 }}
              />
              <Tab 
                icon={<Analytics />} 
                label="Performance" 
                iconPosition="start"
                sx={{ gap: 1 }}
              />
              <Tab 
                icon={<Settings />} 
                label="Calibration" 
                iconPosition="start"
                sx={{ gap: 1 }}
              />
              <Tab 
                icon={<Timeline />} 
                label="Adaptive Analysis" 
                iconPosition="start"
                sx={{ gap: 1 }}
              />
              <Tab 
                icon={<Task />} 
                label="Task Browser" 
                iconPosition="start"
                sx={{ gap: 1 }}
              />
              <Tab 
                icon={<Gavel />} 
                label="Judge Analysis" 
                iconPosition="start"
                sx={{ gap: 1 }}
              />
              <Tab 
                icon={<Terminal />} 
                label="Real-Time Logs" 
                iconPosition="start"
                sx={{ gap: 1 }}
              />
              <Tab 
                icon={<Storage />} 
                label="Data Manager" 
                iconPosition="start"
                sx={{ gap: 1 }}
              />
              <Tab 
                icon={<Psychology />} 
                label="AI Agents" 
                iconPosition="start"
                sx={{ gap: 1 }}
              />
            </Tabs>

            <TabPanel value={tabValue} index={0}>
              <Dashboard />
            </TabPanel>
            <TabPanel value={tabValue} index={1}>
              <Performance />
            </TabPanel>
            <TabPanel value={tabValue} index={2}>
              <Calibration />
            </TabPanel>
            <TabPanel value={tabValue} index={3}>
              <AdaptiveAnalysis />
            </TabPanel>
            <TabPanel value={tabValue} index={4}>
              <TaskBrowser />
            </TabPanel>
            <TabPanel value={tabValue} index={5}>
              <JudgeComparison />
            </TabPanel>
            <TabPanel value={tabValue} index={6}>
              <RealTimeLogs />
            </TabPanel>
            <TabPanel value={tabValue} index={7}>
              <DataManager />
            </TabPanel>
            <TabPanel value={tabValue} index={8}>
              <AgentCards />
            </TabPanel>
          </Paper>
        </Container>
      </div>
    </ThemeProvider>
  );
}

export default App;