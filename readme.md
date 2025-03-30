# 🏈 NFL Draft Predictor Dashboard

A modern, interactive web application built with Dash and Plotly that visualizes NFL draft predictions based on college football player statistics and machine learning models.

![Dashboard Preview](https://via.placeholder.com/800x400?text=NFL+Draft+Predictor+Dashboard)

## 📊 Features

- **Player Selection**: Filter and select players by school, position, and other criteria
- **Draft Prediction**: View probability of a player being drafted using our ML models
- **Statistical Analysis**: Interactive visualizations of player performance metrics
- **Player Comparisons**: Compare selected players against historical draft picks
- **Responsive Design**: Optimized for both desktop and mobile viewing

## 🛠️ Technology Stack

- **[Dash](https://dash.plotly.com/)**: Python framework for building analytical web applications
- **[Plotly](https://plotly.com/python/)**: Interactive visualization library
- **[Bootstrap](https://getbootstrap.com/)**: Frontend styling and responsive design
- **[Pandas](https://pandas.pydata.org/)**: Data manipulation and analysis

## 🚀 Deployment Architecture

The application is deployed using a multi-tier architecture that enables secure, public access:

### 1. Local Development Server

The Dash application runs on a local development server, typically on port 8050:

```python
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
```

### 2. VS Code Port Forwarding

Microsoft Visual Studio Code provides port forwarding capabilities that expose the local development server:

1. In VS Code, open the "PORTS" tab in the bottom panel
2. Click "Forward a Port" and enter 8050
3. VS Code creates a tunnel that forwards the local port to a remote URL

### 3. NGrok Integration

[NGrok](https://ngrok.com/) creates a secure tunnel to the locally running web server and provides a public URL:

1. NGrok connects to the VS Code forwarded port
2. It generates a public HTTPS URL (e.g., `https://abc123.ngrok.io`)
3. This URL is accessible from anywhere on the internet

```bash
# Example NGrok command (if running standalone)
ngrok http 8050
```

### 4. GoDaddy Domain Masking

To provide a professional, branded URL, we use [GoDaddy](https://www.godaddy.com/) domain masking:

1. A custom domain (e.g., `draftpredictor.example.com`) is registered with GoDaddy
2. Domain forwarding is configured to point to the NGrok URL
3. URL masking is enabled to maintain the custom domain in the browser address bar

## 🔧 Setup and Configuration

### Local Development

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python dashapp.py
   ```

### Deployment Configuration

1. **VS Code Port Forwarding**:
   - Open the project in VS Code
   - Navigate to the "PORTS" tab
   - Forward port 8050 to make it accessible remotely

2. **NGrok Setup**:
   - Create an account on [NGrok](https://ngrok.com/)
   - Download and install the NGrok client
   - Authenticate with your NGrok auth token:
     ```bash
     ngrok authtoken YOUR_AUTH_TOKEN
     ```
   - Connect NGrok to your forwarded port

3. **GoDaddy Configuration**:
   - Log in to your GoDaddy account
   - Navigate to your domain's DNS settings
   - Set up domain forwarding to your NGrok URL
   - Enable URL masking to maintain your domain in the address bar

## 🔄 Continuous Deployment

For continuous availability:

1. Use a persistent NGrok URL with a paid NGrok account
2. Set up a service to automatically restart the application if it crashes
3. Consider using a process manager like PM2 or Supervisor

## 📝 Usage Notes

- The NGrok URL changes each time NGrok is restarted (on the free plan)
- Update the GoDaddy forwarding settings whenever the NGrok URL changes
- For production use, consider a more permanent hosting solution like Heroku, AWS, or Azure

## 🔒 Security Considerations

- The application uses HTTPS through NGrok for encrypted data transmission
- No sensitive user data is stored in the application
- Consider implementing authentication for production deployments

## 📚 Additional Resources

- [Dash Documentation](https://dash.plotly.com/introduction)
- [NGrok Documentation](https://ngrok.com/docs)
- [GoDaddy Domain Forwarding Help](https://www.godaddy.com/help/forward-my-domain-12123)
- [VS Code Port Forwarding Guide](https://code.visualstudio.com/docs/remote/ssh#_forwarding-a-port-creating-ssh-tunnels)
