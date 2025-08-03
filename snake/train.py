from agent import Agent, plot, scores, mean_scores
from snake_game import SnakeGame

agent = Agent()
game = SnakeGame()

while True:
    state_old = agent.get_state(game)
    final_move = agent.get_action(state_old)
    reward, done, score = game.play_step(final_move)
    state_new = agent.get_state(game)

    agent.train_short_memory(state_old, final_move, reward, state_new, done)
    agent.remember(state_old, final_move, reward, state_new, done)

    if done:
        game.reset()
        agent.n_games += 1
        agent.train_long_memory()

        scores.append(score)
        mean_scores.append(sum(scores)/len(scores))
        plot(scores, mean_scores)

        print('Game:', agent.n_games, 'Score:', score, 'Mean:', mean_scores[-1])
