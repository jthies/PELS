/*
 * =======================================================================================
 *
 *   RACE: Recursicve Algebraic Coloring Engine
 *   Copyright (C) 2019, RRZE, Friedrich-Alexander-Universität Erlangen-Nürnberg
 *   Author: Christie Alappat
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU Affero General Public License as
 *   published by the Free Software Foundation, either version 3 of the
 *   License, or (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU Affero General Public License for more details.
 *
 *   You should have received a copy of the GNU Affero General Public License
 *   along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 * =======================================================================================
 */

#ifndef RACE_CONFIG_H
#define RACE_CONFIG_H

#define RACE_VERBOSITY 1
#define RACE_KERNEL_THREAD_OMP
#define RACE_USE_GAP
#define RACE_PERMUTE_ON_FLY
#ifdef RACE_USE_GAP
    #define RACE_USE_SOA_GRAPH
#endif
/* #undef RACE_HAVE_CPP_17 */
#define RACE_ENABLE_MPI_MPK

#endif
