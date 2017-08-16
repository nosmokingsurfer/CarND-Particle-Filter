/*
 * particle_filter.cpp
 *
 *  Created on: June 21, 2017
 *  Author: Junsheng Fu
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iterator>

#include "particle_filter.h"

using namespace std;

static int NUM_PARTICLES = 100;

/* Set the number of particles.
 * Initialize all particles to first position (based on estimates of
 * x, y, theta and their uncertainties from GPS) and all weights to 1.
 * Add random Gaussian noise to each particle.
 */
void ParticleFilter::init(double x, double y, double theta, double std[]) {

  // create normal distributions for x, y, and theta
  std::normal_distribution<double> N_x(x, std[0]);
  std::normal_distribution<double> N_y(y, std[1]);
  std::normal_distribution<double> N_theta(theta, std[2]);
  std::default_random_engine gen;

  // resize the vectors of particles and weights
  num_particles = NUM_PARTICLES;

  for(int i = 0; i < NUM_PARTICLES; i++){
  Particle temp;
  
    temp.x = N_x(gen);
    temp.y = N_y(gen);
    temp.theta = N_theta(gen);
    temp.weight = 1;
    particles.push_back(temp);
  }

  is_initialized = true;

}


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    default_random_engine gen;
    
    for(int i = 0; i < num_particles; i++)
    {
       double new_x = 0;
       double new_y = 0;
       double new_theta = 0;
       
       if(fabs(yaw_rate) < 0.0001)
       {
            new_x = particles[i].x + velocity*delta_t*cos(particles[i].theta);
            new_y = particles[i].y + velocity*delta_t*sin(particles[i].theta);
            new_theta = particles[i].theta;
       }
       else
       {
            new_x = particles[i].x + velocity/yaw_rate*(sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
            new_y = particles[i].y + velocity/yaw_rate*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
            new_theta = particles[i].theta + yaw_rate*delta_t;
       }
       
       normal_distribution<double> N_x(new_x, std_pos[0]);
       normal_distribution<double> N_y(new_y, std_pos[1]);
       normal_distribution<double> N_theta(new_theta, std_pos[2]);
       
       particles[i].x = N_x(gen);
       particles[i].y = N_y(gen);
       particles[i].theta = N_theta(gen);
    }
}


void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    for(int i = 0; i < observations.size(); i++)
    {
        int index{-1};
        
        double minError = std::numeric_limits<float>::max();
        
        for(int j = 0; j < predicted.size(); j++)
        {
            double dx = predicted[j].x - observations[i].x;
            double dy = predicted[j].y - observations[i].y;
            double error = sqrt(dx*dx + dy*dy);
        
            if(error < minError)
            {
                minError = error;
                observations[i].id = predicted[j].id;
            }
        }
    }
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {

	for(int i = 0; i < this->num_particles; i++)
	{
 
	    //take coordinates of each particle
	    double px = particles[i].x;
	    double py = particles[i].y;
	    double theta = particles[i].theta;
	    
	    vector<LandmarkObs> landmarksInRange;
	    vector<LandmarkObs> mapObservations;
	    
	    //transform every observation from particle related frame to global frame
	    for(int j = 0; j < observations.size(); j++)
	    {
	        double vr_x = observations[j].x;
	        double vr_y = observations[j].y;
	        
	        LandmarkObs temp;
	        
	        temp.x = px + vr_x*cos(theta) - vr_y*sin(theta);
	        temp.y = py + vr_x*sin(theta) + vr_y*cos(theta);
	        temp.id = observations[j].id;
	        mapObservations.push_back(temp);
	    }
	    	    
	    //find landmarks inside the sensor range
	    for(int j = 0; j < map_landmarks.landmark_list.size(); j++)
	    {
	        double dx = px - map_landmarks.landmark_list[j].x_f;
	        double dy = py - map_landmarks.landmark_list[j].y_f;
	        
	        if(sqrt(dx*dx + dy*dy) < sensor_range)
	        {
	            LandmarkObs temp = {map_landmarks.landmark_list[j].id_i, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f};
	            landmarksInRange.push_back(temp);
	        }
	    }
    
	    //make association between observations and actual landmarks
	    this->dataAssociation(landmarksInRange, mapObservations);
	    
	    
	    particles[i].weight = 1;
	    
	    //update weight for each particle
	    for(int j = 0; j < mapObservations.size(); j++)
	    {
	        int observation_index = mapObservations[j].id;

	        double prediction_x = map_landmarks.landmark_list[observation_index-1].x_f;
	        //cout << prediction_x << endl;
	
	        double prediction_y = map_landmarks.landmark_list[observation_index-1].y_f;
	        //cout << prediction_y << endl;
	        
	        double x_ = pow(mapObservations[j].x - prediction_x,2)/(2*pow(std_landmark[0],2));
	        //cout << "x_ = " << x_ << endl;
	        double y_ = pow(mapObservations[j].y - prediction_y,2)/(2*pow(std_landmark[1],2));
	        //cout << "y_ = " << y_ << endl;
	        double r = exp(-(x_ +y_)) / (2*M_PI*std_landmark[0]*std_landmark[1]);
	        //cout << "r = " << r << endl;
	        particles[i].weight *= r;
	    }
	    
	    this->weights.push_back(particles[i].weight);
	}
}


void ParticleFilter::resample() {
      vector<Particle> resampled_particles;

  random_device rdevice;
  mt19937 gen(rdevice());

  discrete_distribution<int> index(this->weights.begin(), this->weights.end());

  for (int c = 0; c < this->num_particles; c++) {

    const int i = index(gen);

    Particle p {
      i,
      this->particles[i].x,
      this->particles[i].y,
      this->particles[i].theta,
      this->particles[i].weight
    };

    resampled_particles.push_back(p);
  }

  this->particles = resampled_particles;
  this->weights.clear();
}


/* particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
 * associations: The landmark id that goes along with each listed association
 * sense_x: the associations x mapping already converted to world coordinates
 * sense_y: the associations y mapping already converted to world coordinates
 */
Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
