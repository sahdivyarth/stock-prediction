import React from 'react';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import Box from '@mui/material/Box';
import { Tooltip, Chip } from '@mui/material';
import PsychologyIcon from '@mui/icons-material/Psychology';
import TimelineIcon from '@mui/icons-material/Timeline';
import AnalyticsIcon from '@mui/icons-material/Analytics';

export const Header = () => {
  return (
    <AppBar 
      position="static" 
      sx={{ 
        mb: 4,
        background: 'linear-gradient(45deg, #1e3c72 30%, #2a5298 90%)',
        boxShadow: '0 3px 5px 2px rgba(33, 203, 243, .3)',
      }}
    >
      <Toolbar>
        <ShowChartIcon sx={{ mr: 2, fontSize: 28 }} />
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          Stock Market Prediction Model
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <Tooltip title="LSTM-based price prediction" arrow>
            <Chip
              icon={<PsychologyIcon />}
              label="Deep Learning"
              variant="outlined"
              sx={{ 
                color: 'white',
                borderColor: 'rgba(255, 255, 255, 0.5)',
                '& .MuiChip-icon': { color: 'white' }
              }}
            />
          </Tooltip>
          
          <Tooltip title="Technical indicators analysis" arrow>
            <Chip
              icon={<TimelineIcon />}
              label="Technical Analysis"
              variant="outlined"
              sx={{ 
                color: 'white',
                borderColor: 'rgba(255, 255, 255, 0.5)',
                '& .MuiChip-icon': { color: 'white' }
              }}
            />
          </Tooltip>
          
          <Tooltip title="Advanced metrics and visualization" arrow>
            <Chip
              icon={<AnalyticsIcon />}
              label="Real-time Analytics"
              variant="outlined"
              sx={{ 
                color: 'white',
                borderColor: 'rgba(255, 255, 255, 0.5)',
                '& .MuiChip-icon': { color: 'white' },
                display: { xs: 'none', md: 'flex' }
              }}
            />
          </Tooltip>
        </Box>
      </Toolbar>
    </AppBar>
  );
};
