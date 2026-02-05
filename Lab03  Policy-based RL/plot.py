import pandas as pd
import matplotlib.pyplot as plt
import os

class Plotter:
    def __init__(self, csv1, csv2, label1="Run 1", label2="Run 2", path = "./figures/actorcritic/"):
        self.df1 = pd.read_csv(csv1, comment="#")
        self.df2 = pd.read_csv(csv2, comment="#")
        self.label1 = label1
        self.label2 = label2
        self.path = path

    def plot_reward_comparison(self):
        plt.figure()
        print(self.df1["r"])
        plt.plot(self.df1["r"], label=self.label1)
        plt.plot(self.df2["r"], label=self.label2)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward comparison")
        plt.legend()
        plt.grid()
        plt.savefig(f"{self.path}/rewards_comparison.png")
        plt.show()
        plt.close()

    def plot_cumulative_time(self):
        plt.figure()
        plt.plot(self.df1["t"], label=self.label1)
        plt.plot(self.df2["t"], label=self.label2)
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Time (s)")
        plt.title("Cumulative time comparison")
        plt.legend()
        plt.grid()
        plt.savefig(f"{self.path}/cumulative_time.png")
        plt.show()
        plt.close()

    def plot_reward_vs_time(self):
        plt.figure()
        plt.plot(self.df1["t"], self.df1["r"], label=self.label1)
        plt.plot(self.df2["t"], self.df2["r"], label=self.label2)
        plt.xlabel("Time (s)")
        plt.ylabel("Reward")
        plt.title("Reward vs Time")
        plt.legend()
        plt.grid()
        plt.savefig(f"{self.path}/rewards_vs_time.png")
        plt.show()
        plt.close()

    def plot_all(self):
        self.plot_cumulative_time()
        self.plot_reward_comparison()
        self.plot_reward_vs_time()


if __name__=="__main__":
    os.makedirs("./figures/actorcritic/", exist_ok=True)
    plot = Plotter(csv1= "tmp/gym/ppo/monitor.csv", label1= "PPO", csv2= "tmp/gym/sac/monitor.csv", label2= "SAC", path = "./figures/actorcritic/")
    plot.plot_all()
