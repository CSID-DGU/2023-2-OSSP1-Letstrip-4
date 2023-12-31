import React from 'react';
import { Outlet } from 'react-router-dom';
import styled from 'styled-components';
import { BsFacebook, BsYoutube, BsInstagram } from "react-icons/bs";
import { Swiper, SwiperSlide } from "swiper/react";
import { Autoplay} from "swiper";
import 'swiper/css';
import 'swiper/css/pagination';
import 'swiper/css/navigation';
import 'swiper/css/scrollbar';

const FooterWrapper = styled.div`
  background: ${props => props.theme.main};
  height: 200px;
  width: 100%;
  position: relative;
  transform: translateY(100%);
`;
const NoticeWrapper = styled.div`
  border-bottom: 1px solid #717171;
  border-top: 1px solid #717171;  
`;
const NoticeInner = styled.div`
  width: 1024px;
  margin: 0 auto;
  display: flex;
  align-items: center;
  .slide {
    color: #d9d9d9;
  }
`;
const Notice = styled.div`
  padding: 10px;
  margin: 0 auto;
  .notice {
    width: 100px;
    font-size: 20px;
    color: #d9d9d9;
  }
`;
const Mainarea = styled.div`
  width: 1024px;
  margin: 0 auto;
`;
const Menu = styled.div`
  margin-top: 10px;
  .menu {
    display: flex;
    justify-content: space-evenly;
  }
  .footmenu {
    text-decoration: none;
    color: gray;
    &:hover {
      color: black;
      transition: 1s ease-in-out;
    }
  }
`;

const Icons = styled.div`
  color: #717171;
  font-size: 20px;
  margin-top: 10px;
  .icon {
    color: #717171;
    text-decoration: none;
    cursor: pointer;
    margin-right: 10px;
  }
`;
const Infoarea = styled.div`
  margin-top: 10px;
`;
const List = styled.p`
  font-size: 12px;
  line-height: 20px;;
`;
const Submenu = styled.span`
  color: #717171;
  margin-right: 6px;
`;

function Footer(props) {
  return (
    <>
      <FooterWrapper>
        <NoticeWrapper>
          <NoticeInner>
            <Notice>
              <h2 className='notice'>공지사항</h2>
            </Notice>
            <Swiper
              className='swiper'
              modules={[Autoplay]}
              spaceBetween={50}
              slidesPerView={1}
              autoplay={{ delay: 5000 }}
            >
              <SwiperSlide  className='slide'>고객센터 여름휴가 안내</SwiperSlide>
              <SwiperSlide  className='slide'>[TOP 10] 영화 안내</SwiperSlide>
              <SwiperSlide  className='slide'>Movie Green 이용약관 변경 안내</SwiperSlide>
              <SwiperSlide  className='slide'>[직원 채용] 하반기 채용 예정 안내</SwiperSlide>

            </Swiper>
          </NoticeInner>          
        </NoticeWrapper>

        <Mainarea>        
          <Menu>
            <ul className='menu'>
              <li><a className='footmenu' href='/'>회사소개</a></li>
              <li><a className='footmenu' href='/'>인재채용</a></li>
              <li><a className='footmenu' href='/'>서비스 소개</a></li>
              <li><a className='footmenu' href='/'>이용약관</a></li>
              <li><a className='footmenu' href='/'>개인정보 처리방침</a></li>
              <li><a className='footmenu' href='/'>고객센터</a></li>
            </ul>

            <Infoarea>
              <List>
                <Submenu>대표이사: KIM JI SOO</Submenu>
                <Submenu>사업자정보확인</Submenu>
                <Submenu>사업자등록번호:188-18-18198</Submenu>
                <Submenu>통신판매신고번호:2023-인천남동-204호</Submenu>
              </List>
              <List>
                <Submenu>사업장:인천광역시 남동구 문화로 155번길</Submenu>
                <Submenu>호스팅사업자:그린아트구월(주)</Submenu>
              </List>
              <List>
                <Submenu>대표메일:greenMovie@naver.com</Submenu>
                <Submenu>고객센터: 1566-1566(평일 14시~19시, 공휴일 휴무)</Submenu>
              </List>
              <List>
                <Submenu>Copyright©  2023 Incheon GreenMovie All Rights Reserved.</Submenu>
              </List>            
            </Infoarea>

            <Icons>
              <a href='https://ko-kr.facebook.com/' className='icon' ><BsFacebook  /></a>
              <a href='https://www.youtube.com/' className='icon' ><BsYoutube  /></a>
              <a href='https://www.instagram.com/' className='icon'><BsInstagram  /></a>
            </Icons>

          </Menu>
        </Mainarea>
      </FooterWrapper>
    
      <Outlet />
    </>

  );
}

export default Footer;