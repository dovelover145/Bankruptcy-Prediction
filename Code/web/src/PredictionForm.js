import React, { useState } from 'react';
import { Container, TextField, Button, Typography, Box } from '@mui/material';
import axios from 'axios';

const PredictionForm = () => {
  const [inputs, setInputs] = useState(Array(5).fill(''));
  const [prediction, setPrediction] = useState('');
  const [submitted, setSubmitted] = useState(false);

  const handleChange = (index, event) => {
    const newInputs = [...inputs];
    newInputs[index] = event.target.value;
    setInputs(newInputs);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      const response = await axios.post('http://localhost:5000/predict', { inputs });
      setPrediction(response.data.prediction);
      setSubmitted(true);
    } catch (error) {
      console.error('Error submitting prediction request:', error);
    }
  };

  return (
    <Container>
      {submitted ? (
        <Box>
          <Typography variant="h4" gutterBottom>Prediction Result</Typography>
          <Typography variant="h5" color="secondary">Prediction: {prediction}</Typography>
          <Typography variant="h6" >0: The company will stay alive</Typography>
          <Typography variant="h6" >1: The company will go bankrupt</Typography>
        </Box>
      ) : (
        <Box>
          <Typography variant="h4" gutterBottom>Bankruptcy Prediction</Typography>
          <form onSubmit={handleSubmit}>
            {inputs.map((input, index) => (
              <TextField
                key={index}
                label={`Input ${index + 1}`}
                variant="outlined"
                value={input}
                onChange={(event) => handleChange(index, event)}
                fullWidth
                margin="normal"
                type="text"
              />
            ))}
            <Button type="submit" variant="contained" color="primary">Submit</Button>
          </form>
        </Box>
      )}
    </Container>
  );
};

export default PredictionForm;
