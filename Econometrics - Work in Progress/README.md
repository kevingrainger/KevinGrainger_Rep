
These two projects are based of basically the same risk/volitilty measures. i really like Hiddem Markov Model stuff and 
wantred to practive using it in a financial setting. 
The first File is a volitility measuring tool for use on a timeseries. It is very generic and I have not maybe any finance specific
alteration. It is a HMM and GARCH combination, where the GARCH result informs the probaility of emission/change of state in the HMM. 
I don't have much practice on with HMM's so I wanted to use oil market data, as its' spikes coincide with geo-political events, so 
I could gauge the effectivness roughly. 
The HMM works off a forward and backward algorithm as usual. The states assigned, where just low, medium , high volitility. 
It doesn't worl overly well, it need alot of training data. I found it hard to get the model sensitivity right, but it was a fun 
program.

The second program uses this same HMM-GARCH volitility program as a measure for risk in renewable and oil assets. 
It uses modern portfolio theory to manage a 10,000 euro energy portfolio, I only gave it two asset categories to balence. 
Below: The Volitlity measures of the energy assests. 
Below [2] : The porffolio allocation.

<p align="center">
  <img src="Energy Portfolio Optimisations.png" 
       alt="Energy Portfolio Optimisations" width="350" style="margin: 10px;">
  <img src="Hidden Markov Model Oil Volatility Measure.png" 
       alt="Hidden Markov Model Oil Volatility Measure" width="350" style="margin: 10px;">
</p>

