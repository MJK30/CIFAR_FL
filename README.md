# Federated Learning (FL)

Federated Learning (FL) is a technique that allows a single machine to learn from many different sources by converting the data into small pieces and sending them to different FL nodes. It is a decentralized machine learning paradigm that enables model training across various devices while preserving data privacy.

## Centralized FL
- **Coordination** - Involves a central server coordinating the training, aggregation, and distribution process.
- **Aggregation** - Model updates are aggregated at a single point, simplifying the update process.
- **Control** - The central server has control over the training process.

## Cifar.py
- Import the CIFAR-10 dataset.
- Partition the data according to the required number of silos.
- Preprocess the data for training and testing.
- Create training and testing datasets.
- Define the Convolutional Neural Network for training.

## Client.py
- Define and initialize clients using the Flower Framework.
- Create methods to set and get the weights and bias parameters from the CNN model.
- Define fit and evaluate methods to train and test the CNN model using new parameters.

## Server.py
- Create a centralized server for the global model and client coordination.
- Create a strategy to update the model parameters from the clients – FedAvg.

## Steps to Implement
1. Start the centralized server – `server.py`.
2. Run the client according to the required number of silos – `client.py`.
\begin{enumerate}
    \item Start the centralized server – \texttt{server.py}.
    \item Run the client according to the required number of silos – \texttt{client.py}.
\end{enumerate}

\end{document}
