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
  Grid,
  Divider,
  LinearProgress,
  Tooltip,
  Avatar,
} from '@mui/material';
import {
  Psychology,
  TrendingUp,
  Speed,
  CheckCircle,
  Error,
  Analytics,
  Refresh,
} from '@mui/icons-material';

interface AgentData {
  agent_id: string;
  adaptive_performance: {
    final_ability: number;
    confidence_interval: [number, number];
    converged: boolean;
    tasks_completed: number;
    convergence_step?: number;
  };
  static_performance: {
    overall_score: number;
    total_tasks: number;
    category_scores: Record<string, any>;
  };
  comparison_metrics: {
    efficiency_gain: number;
    tasks_saved: number;
    convergence_achieved: boolean;
  };
}

const AgentCards: React.FC = () => {
  const [agents, setAgents] = useState<string[]>([]);
  const [agentData, setAgentData] = useState<Record<string, AgentData>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadAgents = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Get list of agents
      const agentsResponse = await fetch('http://localhost:8001/api/agents/list');
      const agentsData = await agentsResponse.json();
      
      const agentList = agentsData.agents || [];
      setAgents(agentList);
      
      // If no agents found, try to get from adaptive data directly
      if (agentList.length === 0) {
        const adaptiveResponse = await fetch('http://localhost:8001/api/data/adaptive_results');
        const adaptiveData = await adaptiveResponse.json();
        
        if (adaptiveData.adaptive_evaluation_results) {
          // Create a mock agent for the adaptive results
          setAgents(['adaptive_agent_1']);
          
          const mockAgentData: AgentData = {
            agent_id: 'adaptive_agent_1',
            adaptive_performance: {
              final_ability: adaptiveData.adaptive_evaluation_results.final_ability_estimate || 0,
              confidence_interval: adaptiveData.adaptive_evaluation_results.ability_confidence_interval || [0, 0],
              converged: adaptiveData.adaptive_evaluation_results.convergence_achieved || false,
              tasks_completed: adaptiveData.adaptive_evaluation_results.total_items_administered || 0,
              convergence_step: adaptiveData.adaptive_evaluation_results.total_items_administered,
            },
            static_performance: {
              overall_score: 0.85, // Mock score
              total_tasks: 9, // Based on typical evaluation
              category_scores: {},
            },
            comparison_metrics: {
              efficiency_gain: adaptiveData.adaptive_evaluation_results.total_items_administered ? 
                ((9 - adaptiveData.adaptive_evaluation_results.total_items_administered) / 9) * 100 : 0,
              tasks_saved: 9 - (adaptiveData.adaptive_evaluation_results.total_items_administered || 0),
              convergence_achieved: adaptiveData.adaptive_evaluation_results.convergence_achieved || false,
            },
          };
          
          setAgentData({ 'adaptive_agent_1': mockAgentData });
        }
      } else {
        // Load performance data for each agent
        const agentDataPromises = agentList.map(async (agentId: string) => {
          try {
            const response = await fetch(`http://localhost:8001/api/agents/${agentId}/performance`);
            const data = await response.json();
            return { agentId, data };
          } catch (err) {
            console.error(`Failed to load data for agent ${agentId}:`, err);
            return { agentId, data: null };
          }
        });
        
        const results = await Promise.all(agentDataPromises);
        const agentDataMap: Record<string, AgentData> = {};
        
        results.forEach(({ agentId, data }) => {
          if (data) {
            agentDataMap[agentId] = data;
          }
        });
        
        setAgentData(agentDataMap);
      }
    } catch (err) {
      setError('Failed to load agent data');
      console.error('Agent load error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAgents();
  }, []);

  const getAgentAvatar = (agentId: string) => {
    const colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336'];
    const index = agentId.length % colors.length;
    return colors[index];
  };

  const getPerformanceColor = (score: number) => {
    if (score >= 0.8) return '#4CAF50';
    if (score >= 0.6) return '#FF9800';
    return '#F44336';
  };

  const formatAbility = (ability: number) => {
    if (Math.abs(ability) > 1000) {
      return `${(ability / 1000).toFixed(1)}k`;
    }
    return ability.toFixed(2);
  };

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

  if (agents.length === 0) {
    return (
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <Alert severity="info" sx={{ mb: 2 }}>
          No agent data available. Run an adaptive evaluation to see agent performance cards.
        </Alert>
        <Button 
          variant="contained" 
          onClick={loadAgents}
          startIcon={<Refresh />}
        >
          Refresh Agent Data
        </Button>
      </Box>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
        <Typography variant="h2" sx={{ 
          background: 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
        }}>
          AI Agent Performance
        </Typography>
        <Button 
          variant="outlined" 
          onClick={loadAgents}
          startIcon={<Refresh />}
          size="small"
        >
          Refresh
        </Button>
      </Box>

      <Grid container spacing={3}>
        {agents.map((agentId) => {
          const agent = agentData[agentId];
          if (!agent) return null;

          return (
            <Grid item xs={12} md={6} lg={4} key={agentId}>
              <Card sx={{ 
                height: '100%',
                background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
                border: '1px solid #e0e0e0',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: '0 8px 25px rgba(0,0,0,0.1)',
                },
                transition: 'all 0.3s ease',
              }}>
                <CardContent>
                  {/* Agent Header */}
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Avatar sx={{ 
                      bgcolor: getAgentAvatar(agentId), 
                      mr: 2,
                      width: 48,
                      height: 48,
                    }}>
                      <Psychology />
                    </Avatar>
                    <Box>
                      <Typography variant="h6" fontWeight="bold">
                        {agentId.replace('_', ' ').toUpperCase()}
                      </Typography>
                      <Chip
                        label={agent.adaptive_performance.converged ? 'Converged' : 'In Progress'}
                        size="small"
                        color={agent.adaptive_performance.converged ? 'success' : 'warning'}
                        icon={agent.adaptive_performance.converged ? <CheckCircle /> : <Error />}
                      />
                    </Box>
                  </Box>

                  <Divider sx={{ mb: 2 }} />

                  {/* Adaptive Performance */}
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle2" color="textSecondary" gutterBottom>
                      Adaptive Performance
                    </Typography>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2">Ability Estimate:</Typography>
                      <Typography variant="body2" fontWeight="bold">
                        {formatAbility(agent.adaptive_performance.final_ability)}
                      </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2">Tasks Completed:</Typography>
                      <Typography variant="body2" fontWeight="bold">
                        {agent.adaptive_performance.tasks_completed}
                      </Typography>
                    </Box>
                    {agent.adaptive_performance.convergence_step && (
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="body2">Convergence Step:</Typography>
                        <Typography variant="body2" fontWeight="bold">
                          {agent.adaptive_performance.convergence_step}
                        </Typography>
                      </Box>
                    )}
                  </Box>

                  {/* Efficiency Metrics */}
                  {agent.comparison_metrics && (
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="subtitle2" color="textSecondary" gutterBottom>
                        Efficiency Metrics
                      </Typography>
                      <Box sx={{ mb: 1 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                          <Typography variant="body2">Efficiency Gain:</Typography>
                          <Typography variant="body2" fontWeight="bold" color="success.main">
                            {agent.comparison_metrics.efficiency_gain.toFixed(1)}%
                          </Typography>
                        </Box>
                        <LinearProgress
                          variant="determinate"
                          value={Math.min(agent.comparison_metrics.efficiency_gain, 100)}
                          sx={{ height: 6, borderRadius: 3 }}
                          color="success"
                        />
                      </Box>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="body2">Tasks Saved:</Typography>
                        <Tooltip title="Compared to static evaluation">
                          <Chip
                            label={`${agent.comparison_metrics.tasks_saved} tasks`}
                            size="small"
                            color="primary"
                            icon={<Speed />}
                          />
                        </Tooltip>
                      </Box>
                    </Box>
                  )}

                  {/* Action Buttons */}
                  <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
                    <Button
                      size="small"
                      variant="outlined"
                      startIcon={<Analytics />}
                      fullWidth
                    >
                      View Details
                    </Button>
                    <Button
                      size="small"
                      variant="contained"
                      startIcon={<TrendingUp />}
                      fullWidth
                    >
                      Trajectory
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          );
        })}
      </Grid>

      {/* Summary Stats */}
      <Card sx={{ mt: 3, background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', color: 'white' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Agent Performance Summary
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={4}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h4" fontWeight="bold">
                  {agents.length}
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9 }}>
                  Total Agents
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h4" fontWeight="bold">
                  {Object.values(agentData).filter(a => a.adaptive_performance.converged).length}
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9 }}>
                  Converged
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h4" fontWeight="bold">
                  {Object.values(agentData).reduce((avg, agent) => 
                    avg + (agent.comparison_metrics?.efficiency_gain || 0), 0) / agents.length || 0}%
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9 }}>
                  Avg Efficiency
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Box>
  );
};

export default AgentCards; 