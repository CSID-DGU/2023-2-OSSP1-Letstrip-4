import React from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { selectLogin, selectUser, selectUserName } from '../feature/user/userSlice';
import styled from 'styled-components';

const NavWrapper = styled.div`
  .navBar {
    padding: 10px;
    cursor: pointer;
    &:hover{
      color: #586F8C;
      transition: 1s ease-in-out;
    }
  }
`;

function LogoutBar(props) {
  const userName = useSelector(selectUserName);
  const dispatch = useDispatch();

  const handleLogout = () => {
    dispatch(selectLogin(false));
    dispatch(selectUser(''));
  };

  return (
    <>
      <NavWrapper>

        <a className="navBar" onClick={undefined}>{userName}님</a>
        <a className="navBar" onClick={handleLogout}>로그아웃</a>

      </NavWrapper>
    </>
  );
}

export default LogoutBar;