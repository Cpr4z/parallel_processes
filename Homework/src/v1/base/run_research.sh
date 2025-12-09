#!/bin/bash
#SBATCH --job-name="homework_base"
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --time=0-00:20:00
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1992

matrix_size=(500 1000 1500 2000 2500 3000)

printf "" > result.txt
for board_size in "${matrix_size[@]}"
do
	./main.exe $board_size >> result.txt
	
    printf "\n\n\n" >> result.txt
done