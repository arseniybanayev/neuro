using System;
using System.Collections.Generic;
using System.Linq;

namespace Neuro
{
	internal class Neuron
	{
		private readonly List<Synapse> _inputSynapses = new List<Synapse>();
		private readonly List<Synapse> _outputSynapses = new List<Synapse>();

		private Lazy<double> _output;

		internal double Output {
			get { return _output.Value; }
			set {
				if (_inputSynapses.Any()) // Output can only be set manually on input neurons
					throw new Exception("Cannot set output on a neuron in a non-input layer");
				_output = new Lazy<double>(() => value);
				ResetDependentOutputs();
			}
		}

		private void ResetDependentOutputs() {
			// TODO: Replace Lazy<double> with something that tracks dependencies and marks itself "dirty"
			foreach (var outputSynapse in _outputSynapses) {
				outputSynapse.OutputNeuron.ResetOutput();
				outputSynapse.OutputNeuron.ResetDependentOutputs();
			}
		}

		private void ResetOutput() {
			_output = new Lazy<double>(() => {
				if (_inputSynapses.Count == 0) // Output has to be set manually for input neurons
					throw new Exception("The output for this input-layer neuron has not been set");

				// logistic_sigmoid(weighted_sum_of_inputs)
				return NMath.Sigmoid(_inputSynapses.Sum(s => s.Weight * s.InputNeuron.Output));
			});
		}

		internal Neuron(IEnumerable<Neuron> inputNeurons = null) {
			foreach (var inputNeuron in inputNeurons ?? Enumerable.Empty<Neuron>()) {
				var synapse = new Synapse(inputNeuron, this);
				_inputSynapses.Add(synapse);
				inputNeuron._outputSynapses.Add(synapse);
			}

			ResetOutput();
		}

		public double CalculateError(double target) => Math.Pow(Output - target, 2);

		private Lazy<double> _delta;

		internal double Delta {
			get { return _delta.Value; }
			set {
				if (_outputSynapses.Any()) // Delta can only be set manually on output neurons
					throw new Exception("Cannot set delta on a neuron in a non-output layer");
				_delta = new Lazy<double>(() => value);
				ResetDependentDeltas();
			}
		}

		private void ResetDependentDeltas() {
			// TODO: Replace Lazy<double> with something that tracks dependencies and marks itself "dirty"
			foreach (var inputSynapse in _inputSynapses) {
				inputSynapse.InputNeuron.ResetDelta();
				inputSynapse.InputNeuron.ResetDependentDeltas();
			}
		}

		private void ResetDelta() {
			_delta = new Lazy<double>(() => {
				if (_outputSynapses.Count == 0) // Delta has to be set manually for output neurons
					throw new Exception("The delta for this output-layer neuron has not been set");

				return _outputSynapses.Sum(s => s.OutputNeuron.Delta * s.Weight) * Output * (1 - Output);
			});
		}

		internal void UpdateInputWeights(double learningRate) {
			foreach (var inputSynapse in _inputSynapses) {
				var dWeight = -learningRate * Delta * inputSynapse.InputNeuron.Output;
				inputSynapse.Weight += dWeight;
			}
		}
	}

	internal class Synapse
	{
		internal Neuron InputNeuron { get; }
		internal Neuron OutputNeuron { get; }

		internal Synapse(Neuron inputNeuron, Neuron outputNeuron) {
			InputNeuron = inputNeuron;
			OutputNeuron = outputNeuron;
			Weight = NMath.GetRandomWeight();
		}

		internal double Weight { get; set; }
	}

	public static class NMath
	{
		private static readonly Random Random = new Random();

		public static double GetRandomWeight() => Random.NextDouble();

		public static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

		public static double SigmoidDerivative(double x) => x * (1 - x);
	}

	public class TrainingData
	{
		public IReadOnlyList<double> Inputs { get; }
		public IReadOnlyList<double> Targets { get; }

		public TrainingData(IReadOnlyList<double> inputs, IReadOnlyList<double> targets) {
			Inputs = inputs;
			Targets = targets;
		}
	}

	public class Network
	{
		public IReadOnlyList<double> Run(IReadOnlyList<double> inputs) {
			PropagateForward(inputs);
			return OutputLayer.Select(n => n.Output).ToArray();
		}

		public void Train(IEnumerable<TrainingData> trainingData, int numRounds) {
			var trainingDataArr = trainingData.ToArray();
			for (var i = 0; i < numRounds; i++) {
				foreach (var data in trainingDataArr) {
					PropagateForward(data.Inputs);
					PropagateBackward(data.Targets);
				}
			}
		}

		private const double LearningRate = 0.8;

		private void PropagateBackward(IReadOnlyList<double> targets) {
			if (targets.Count != OutputLayer.Count)
				throw new Exception($"Output layer contains {OutputLayer.Count} neurons but {targets.Count} targets were provided");
			for (var i = 0; i < targets.Count; i++) {
				var output = OutputLayer[i].Output;
				OutputLayer[i].Delta = (output - targets[i]) * output * (1 - output);
				OutputLayer[i].UpdateInputWeights(LearningRate);
			}
			foreach (var layer in HiddenLayers) {
				foreach (var neuron in layer)
					neuron.UpdateInputWeights(LearningRate);
			}
		}

		private void PropagateForward(IReadOnlyList<double> inputs) {
			if (inputs.Count != InputLayer.Count)
				throw new Exception($"Input layer contains {InputLayer.Count} neurons but {inputs.Count} inputs were provided");
			for (var i = 0; i < inputs.Count; i++)
				InputLayer[i].Output = inputs[i];
		}

		private readonly IReadOnlyList<IReadOnlyList<Neuron>> _layers;

		private IReadOnlyList<Neuron> InputLayer => _layers[0];
		private IEnumerable<IReadOnlyList<Neuron>> HiddenLayers => _layers.Skip(1).Take(_layers.Count - 2);
		private IReadOnlyList<Neuron> OutputLayer => _layers[_layers.Count - 1];

		public Network(IReadOnlyList<int> neuronCounts) {
			if (neuronCounts.Count < 3)
				throw new Exception($"{nameof(neuronCounts)} must have at least 3 elements (one input layer, at least one hidden layer and one output layer)");
			var layers = new List<List<Neuron>> {Enumerable.Range(0, neuronCounts[0]).Select(_ => new Neuron()).ToList()};
			for (var i = 1; i < neuronCounts.Count; i++)
				layers.Add(Enumerable.Range(0, neuronCounts[i]).Select(_ => new Neuron(layers[i - 1])).ToList());
			_layers = layers;
		}
	}
}