import React, { useEffect } from 'react';
import useAuthStore from '@/store/authStore';
import { useNavigate } from 'react-router-dom';
import AdminDashboard from './AdminDashboard';
import UserDashboard from './UserDashboard';
import JobProviderDashboard from './JobProviderDashboard';

const Dashboard = () => {
  const { user, logout, setUser } = useAuthStore();
  const navigate = useNavigate();

  useEffect(() => {
    const fetchUser = async () => {
      try {
        const response = await fetch('/api/auth/user', {
          headers: {
            Authorization: `Token ${localStorage.getItem('token')}`,
          },
        });
        if (response.ok) {
          const userData = await response.json();
          setUser(userData);
        } else {
          logout();
          navigate('/login');
        }
      } catch (error) {
        console.error('Failed to fetch user', error);
        logout();
        navigate('/login');
      }
    };

    if (!user) {
      fetchUser();
    }
  }, [user, setUser, logout, navigate]);


  if (!user) {
    return <div>Loading...</div>;
  }

  if (user.role === 'admin') {
    return <AdminDashboard />;
  }

  if (user.role === 'job_provider') {
    return <JobProviderDashboard />;
  }

  return <UserDashboard />;
};

export default Dashboard;
