// Quick test script to verify backend is accessible
fetch('http://localhost:8000/api/v1/system/status')
  .then(res => res.json())
  .then(data => console.log('✓ Backend API accessible:', data))
  .catch(err => console.error('✗ Backend API error:', err));
