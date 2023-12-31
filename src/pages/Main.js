import React from 'react';
import MovieTopNumber from '../components/MovieTopNumber';
import MoviePick from '../components/MoviePick';
import MovieListVote from "../components/MovieListVote";
import { useSelector } from 'react-redux';
import { selectLoginUser } from '../feature/user/userSlice';

function Main(props) {
  const logInStatus = useSelector(selectLoginUser);
  return (
    <div>
      <MovieTopNumber />
      {logInStatus && <MoviePick/>}
      <MovieListVote />
    </div>
  );
}

export default Main;
