import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import useAuthStore from '@/store/authStore';
import { Navigation } from '@/components/Navigation';
import { useNavigate } from 'react-router-dom';

const AdminDashboard = () => {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const { token } = useAuthStore();
  const navigate = useNavigate();

  const handleAddJobProvider = async () => {
    try {
      const response = await fetch('/api/auth/admin/create_job_provider', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Token ${token}`,
        },
        body: JSON.stringify({ username, email, password }),
      });

      if (response.ok) {
        alert('Job provider created successfully');
        setUsername('');
        setEmail('');
        setPassword('');
      } else {
        alert('Failed to create job provider');
      }
    } catch (error) {
      console.error('Error creating job provider:', error);
      alert('An error occurred while creating the job provider');
    }
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <Navigation />
      <div className="flex flex-1 justify-center items-center pt-20">
        <div className="w-full max-w-xs">
          <form className="bg-white shadow-md rounded px-8 pt-6 pb-8 mb-4">
            <h2 className="text-2xl font-bold mb-4">Add Job Provider</h2>
            <div className="mb-4">
              <Label htmlFor="username">
                Username
              </Label>
              <Input
                id="username"
                type="text"
                placeholder="Username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
              />
            </div>
            <div className="mb-4">
              <Label htmlFor="email">
                Email
              </Label>
              <Input
                id="email"
                type="email"
                placeholder="Email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>
            <div className="mb-6">
              <Label htmlFor="password">
                Password
              </Label>
              <Input
                id="password"
                type="password"
                placeholder="******************"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
            <div className="flex items-center justify-between">
              <Button type="button" onClick={handleAddJobProvider}>
                Add Job Provider
              </Button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default AdminDashboard;
