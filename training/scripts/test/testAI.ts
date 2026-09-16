/**
 * Test AI behavior by simulating a full game with the trained NN model.
 *
 * Usage: npx tsx training/scripts/test/testAI.ts
 *        npx tsx training/scripts/test/testAI.ts --games 5        # Run multiple games
 *        npx tsx training/scripts/test/testAI.ts --max-turns 80   # Increase max turns per game
 */
import {setupBackend} from "../../src/setupBackend";
import GameSystem from "../../../src/lib/GameSystem";
import Config from "../../../src/lib/data/Config";
import {executeNNTurn} from "../../../src/AI/TurnExecutor";
import {NNModel} from "../../../src/AI/nn/NNModel";
import {nodeFileSystem} from "../../src/nodeIO";
import {randomTurn} from "../../src/Opponents";
import * as path from "path";

function parseArgs(): {numGames: number; maxTurns: number} {
  const args = process.argv.slice(2);
  let numGames = 1;
  let maxTurns = 100;

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case "--games":
        numGames = parseInt(args[++i]);
        break;
      case "--max-turns":
        maxTurns = parseInt(args[++i]);
        break;
    }
  }

  return {numGames, maxTurns};
}

async function runGame(model: NNModel, maxTurns: number, gameIndex: number): Promise<{
  winner: string | null;
  turns: number;
}> {
  const game = new GameSystem(Config);

  console.log(`\n${"═".repeat(60)}`);
  console.log(`  Game ${gameIndex + 1}`);
  console.log(`${"═".repeat(60)}`);

  for (let turn = 0; turn < maxTurns * 3 && !game.gameOver; turn++) {
    const player = game.currentPlayer;
    const playerIdx = game.currentPlayerIndex;

    if (player.defeated) {
      game.endTurn();
      continue;
    }

    // Blue = NN, Red/Green = random opponent
    if (player.name === "Blue") {
      executeNNTurn(game, model);
    } else {
      randomTurn(game);
    }

    // Print state after each player's turn (every 3rd turn = 1 round)
    if (playerIdx === 2 || game.winner) {
      const p = game.players;
      const playerInfos = p.map(pl => {
        if (pl.defeated) return `${pl.name}(DEAD)`;
        const units = pl.armies.reduce((s, a) => s + a.units.length, 0);
        const armyDetails = pl.armies
          .map(a => `${a.units.length}${a.unitType[0]}@${a.location.replace(/ /g, "")}`)
          .join(",");
        return `${pl.name}($${pl.money},${units}u): ${armyDetails || "none"}`;
      });

      const nodeCount = p.map(pl => {
        let count = 0;
        for (const [, owner] of game.nodeOwnership) {
          if (owner === pl) count++;
        }
        return `${pl.name[0]}=${count}`;
      });

      console.log(`  Turn ${game.turnCount} | ${nodeCount.join(",")}`);
      for (const info of playerInfos) {
        console.log(`    ${info}`);
      }
    }
  }

  if (game.winner) {
    console.log(`\n  >>> ${game.winner.name} WINS! (Turn ${game.turnCount}) <<<`);
    return {winner: game.winner.name, turns: game.turnCount};
  } else {
    console.log(`\n  >>> Game timed out at turn ${game.turnCount} <<<`);
    return {winner: null, turns: game.turnCount};
  }
}

async function main() {
  await setupBackend();

  const {numGames, maxTurns} = parseArgs();

  // Load model
  const modelDir = path.resolve("public/model");
  const model = new NNModel();

  try {
    await model.load(nodeFileSystem(modelDir));
    console.log("Model loaded.\n");
  } catch (e) {
    console.error("Failed to load model. Publish one first: npx tsx training/scripts/publishModel.ts <phase1|phase2|phase3>");
    console.error(e);
    process.exit(1);
  }

  const results: {winner: string | null; turns: number}[] = [];

  for (let g = 0; g < numGames; g++) {
    const result = await runGame(model, maxTurns, g);
    results.push(result);
  }

  // Summary
  if (numGames > 1) {
    console.log(`\n${"═".repeat(60)}`);
    console.log("  Summary");
    console.log(`${"═".repeat(60)}`);

    const wins: Record<string, number> = {};
    let draws = 0;
    let totalTurns = 0;

    for (const r of results) {
      totalTurns += r.turns;
      if (r.winner) {
        wins[r.winner] = (wins[r.winner] || 0) + 1;
      } else {
        draws++;
      }
    }

    for (const [name, count] of Object.entries(wins)) {
      console.log(`  ${name}: ${count} wins (${(100 * count / numGames).toFixed(0)}%)`);
    }
    if (draws > 0) {
      console.log(`  Draws: ${draws} (${(100 * draws / numGames).toFixed(0)}%)`);
    }
    console.log(`  Avg turns: ${(totalTurns / numGames).toFixed(1)}`);
  }

  model.dispose();
}

main().catch(console.error);
