using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Neuro;

namespace NeuroTest
{
	[TestClass]
	public class BasicTests
	{
		[TestMethod]
		public void TrainXOR() {
			var trainingData = new[] {
				new TrainingData(new[] {1.0, 1.0}, new[] {0.0}),
				new TrainingData(new[] {1.0, 0.0}, new[] {1.0}),
				new TrainingData(new[] {0.0, 1.0}, new[] {1.0}),
				new TrainingData(new[] {0.0, 0.0}, new[] {0.0})
			};

			var network = new Network(new[] {2, 3, 1});
			var shittyResults = network.Run(trainingData[0].Inputs);
			Console.WriteLine($"Shitty (untrained) results: {string.Join(", ", shittyResults)}");
			network.Train(trainingData, 100000);
			var goodResults = network.Run(trainingData[0].Inputs);
			Console.WriteLine($"Good (trained) results: {string.Join(", ", goodResults)}");
		}
	}
}