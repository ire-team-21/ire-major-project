import React, {useState} from 'react';
import './App.scss';
import {
    Alignment,
    Button,
    Card,
    Classes,
    Divider,
    Drawer,
    DrawerSize, Elevation, Icon,
    Menu, MenuItem,
    Navbar,
    Position,
    Switch as SwitchComponent
} from "@blueprintjs/core";
import {BrowserRouter, Link, Route, Switch} from "react-router-dom";
import {Home} from "./home/Home";
import {About} from "./about/About";

function App() {
    const [darkTheme, setDarkTheme] = useState(false);

  return (
    <div className={darkTheme ? 'bp3-dark' : ''}>
        <BrowserRouter>
      <header className="App-header">
          <Navbar>
              <Navbar.Group align={Alignment.LEFT}>
                  <img className="logo" src="https://www.iiit.ac.in/img/iiit-new.png" alt="IIIT Logo"/>
                  <Navbar.Divider />
                  <Navbar.Heading>Profiler</Navbar.Heading>
                  <Navbar.Divider />
                  <Link  to="/">
                      <Button className="bp4-minimal" icon="home" text="Home" />
                  </Link>
              </Navbar.Group>
              <Navbar.Group align={Alignment.RIGHT}>
                  <Link  to="/about">
                      <Icon icon={"info-sign"} size={20}/>
                  </Link>
                  <Navbar.Divider/>
                  <SwitchComponent onChange={() => setDarkTheme(!darkTheme)} labelElement={<strong>Dark Theme</strong>} />
              </Navbar.Group>
          </Navbar>
      </header>
        <main className="main">
            <Card className='card'  elevation={Elevation.FOUR} interactive={true}>
            <Switch>
                <Route exact path="/">
                    <Home/>
                </Route>
                <Route path ="/about">
                    <About/>
                </Route>
            </Switch>
            </Card>
        </main>
        </BrowserRouter>
        <Divider />
        <footer className="footer">
            <div>
                Copyright © 2021,
                International Institute of Information Technology, Hyderabad.
                All rights reserved
            </div>
            <div>
                <a href="https://www.iiit.ac.in/privacy-policy/">Privacy policy</a> |
                <a href="https://www.iiit.ac.in/disclosures/">Disclosure</a> |
                <a href="https://www.iiit.ac.in/naac-report/">NAAC Report</a> |
                <a href="https://www.iiit.ac.in/aicte-report/">AICTE</a> |
                <a href="https://www.iiit.ac.in/nirf-report/">NIRF Report</a> |
                <a href="https://www.iiit.ac.in/ariia/">ARIIA</a> |
                <a href="https://www.iiit.ac.in/contact/">Contact Us</a>
            </div>
        </footer>
    </div>
  );
}

export default App;
