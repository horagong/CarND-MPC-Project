#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"

// for convenience
using json = nlohmann::json;

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

double polyprime_eval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 1; i < coeffs.size(); i++) {
    result += i * coeffs[i] * pow(x, i - 1);
  }
  return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}
// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}



int main() {
  uWS::Hub h;

  // MPC is initialized here!
  MPC mpc;

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object sent back from the simulator command server.

          vector<double> ptsx = j[1]["ptsx"]; // The global x positions of the waypoints.
          // This corresponds to the z coordinate in Unity since y is the up-down direction.
          vector<double> ptsy = j[1]["ptsy"]; // The global y positions of the waypoints. 
          // converted from the Unity format to the standard format expected in most mathemetical functions.(0 at East)
          double psi = j[1]["psi"]; // The orientation of the vehicle in radians 
          // psi_unity (float) - The orientation of the vehicle in radians. 
          //    This is an orientation commonly used in navigation.(0 at North)
          double px = j[1]["x"]; // The global x position of the vehicle.
          double py = j[1]["y"]; // The global y position of the vehicle.
          double delta = j[1]["steering_angle"]; // The current steering angle in radians.
          double a = j[1]["throttle"]; // The current throttle value [-1, 1].
          double v = j[1]["speed"]; // The current velocity in mph.

          /*
          * DONE: Calculate steering angle and throttle using MPC.
          *
          * Both are in between [-1, 1].
          *
          */

          auto local_waypoints = Eigen::MatrixXd(2, ptsx.size());
          for (auto i = 0; i < ptsx.size() ; ++i){
            local_waypoints(0, i) =  cos(-psi) * (ptsx[i] - px) - sin(-psi) * (ptsy[i] - py);
            local_waypoints(1, i) =  sin(-psi) * (ptsx[i] - px) + cos(-psi) * (ptsy[i] - py);  
          } 
          auto coeffs = polyfit(local_waypoints.row(0), local_waypoints.row(1), 3);

          double state_x = 0;
          double state_y = 0;
          double state_psi = 0;
          double cte = polyeval(coeffs, state_x);
          double epsi = state_psi - atan(polyprime_eval(coeffs, state_x));

          Eigen::VectorXd state(6);
          state << state_x, state_y, state_psi, v, cte, epsi;
          auto pred_vars = mpc.Solve(state, coeffs);

          json msgJson;
          // NOTE: Remember to divide by deg2rad(25) before you send the steering value back.
          // Otherwise the values will be in between [-deg2rad(25), deg2rad(25] instead of [-1, 1].
          // In the simulator, a positive value of delta implies a right turn and a negative value implies a left turn.
          delta = -pred_vars[0]/deg2rad(25); 
          a = pred_vars[1];

          msgJson["steering_angle"] = delta;
          msgJson["throttle"] = a;

          //Display the MPC predicted trajectory 
          vector<double> mpc_x_vals;
          vector<double> mpc_y_vals;

          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Green line
          for (int i = 2; i < pred_vars.size(); i++) {
            if (i % 2 == 0) {
              mpc_x_vals.push_back(pred_vars[i]);
              //cout << "[" << i << "]" << " mpc_xy = " << pred_vars[i];
            } else {
              mpc_y_vals.push_back(pred_vars[i]);
              //cout << ", " << pred_vars[i] << endl;
            }
          }
          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["mpc_y"] = mpc_y_vals;

          // Display the waypoints/reference line
          vector<double> next_x_vals;
          vector<double> next_y_vals;

          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Yellow line
          // A singular unity unit in the simulator in equivalent to 1 meter
          for (int i = 0 ; i < ptsx.size(); ++i) {
            double x = local_waypoints(0, i);
            double y = polyeval(coeffs, x);
            next_x_vals.push_back(x);
            next_y_vals.push_back(y);
            //cout << "[" << i << "]" << " next_xy = " << x << ", " << y << endl;
          }
          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          int latency_m = MPC::latency * 1000;
          this_thread::sleep_for(chrono::milliseconds(latency_m));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
