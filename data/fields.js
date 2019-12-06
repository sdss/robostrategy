$.ajaxSetup({
				async: false
});

// create a stage for the robot
var stage = acgraph.create('stage-container', 1000, 1000);
var control = acgraph.create('control-container', 300, 600);
var fieldid = -1;
var iexp = 0;
var irobot = 0;
var plan = 'beta-2-kaiju-2';
var observatory = 'apo';
var field_cadence = "";
var racen = -1.;
var deccen = -1.;

// Axis settings
var mm2pix = 1.3;
var xpix0 = 330 * mm2pix;
var ypix0 = 330 * mm2pix;
var alphaLen = Math.round(7.4 * mm2pix);
var betaLen = Math.round(15 * mm2pix);
var alphaWid = Math.round(1. * mm2pix);
var betaWid = Math.round(3.2 * mm2pix);

var text_targetid	= control.text(0, 220, 'Target ID = ');
text_targetid.fontSize('14px');

var text_robotid	= control.text(0, 250, 'Robot ID = ');
text_robotid.fontSize('14px');

var target_obj = {};
var robot_obj = new Array();
var covers = new Array();
var robots = new Array();
var targets = new Array();

var text_coverage = {};
var text_robots = {};
var	show_coverage = {};
var	show_robots = {};

drawTextCoverage();
drawTextRobots();

// Read in information for the field
function readFieldID(fieldid) {
    $.getJSON('targets/rsFieldAssignments-' + plan + '-' +
		          observatory + '-' + fieldid + '.json', function(jd) {
		    target_obj = jd.target_obj;
		    robot_obj = jd.robot_obj;
				racen = jd.racen;
				deccen = jd.deccen;
				field_cadence = jd.field_cadence;
				nwithin = jd.nwithin;
		}).fail(function() { alert("Error reading field."); });
}

function setExposure(exposure) {
		iexp = exposure;
		if(fieldid >= 0) {
				setAllRobots();
		}
	  setExposureTableInfo();
}

// Set the field ID to draw
function setFieldID(in_fieldid) {
		fieldid = in_fieldid;
    readFieldID(fieldid);
		drawAllTargets();
		drawAllCoverage();
		drawAllRobots();
		setFieldTableInfo();
}

function setFieldTableInfo() {
    document.getElementById("fieldid").innerHTML = fieldid;
    document.getElementById("nexposure").innerHTML = robot_obj.length;
    document.getElementById("nwithin").innerHTML = nwithin;
    document.getElementById("racen").innerHTML = racen;
    document.getElementById("deccen").innerHTML = deccen;
    document.getElementById("field_cadence").innerHTML = field_cadence;
}

function setExposureTableInfo() {
    document.getElementById("exposureid").innerHTML = iexp;
    napogee = 0;
    nboss = 0;
    for (i = 0; i < robot_obj[iexp].xPos.length; i++) {
        if(robot_obj[iexp].isAssigned[i]) {
            if(robot_obj[iexp].fiberID[i] == 1) {
						    napogee = napogee + 1;
            } else {
						    nboss = nboss + 1;
        		}
        }
    }
    document.getElementById("napogee").innerHTML = napogee;
    document.getElementById("nboss").innerHTML = nboss;
}

// mm to pixel units
function xymm2pix(xmm, ymm) {
		xpix = (330 + xmm) * mm2pix;
		ypix = (330 - ymm) * mm2pix;
		return [xpix, ypix];
}

// Remove coverage drawing elements and erase list
function eraseCoverage() {
    ncovers = covers.length;
    for (i = 0; i < ncovers; i++) {
				covers[0].remove();
				covers.shift();
    }
}

// Draw an individual coverage donut, and return graphics element
function drawCoverage(robot_obj, i) {
		var [xpix, ypix] = xymm2pix(robot_obj.xPos[i], robot_obj.yPos[i]);
		var cover = stage.donut(xpix, ypix, betaLen - alphaLen,
		                        betaLen + alphaLen, 0, 360);
		if(robot_obj.hasApogee[i]) {
				cover.fill('red', 0.2);
    } else {
				cover.fill('blue', 0.2);
	  }
		return cover;
}

// Draw all coverage donuts (erase any existing). 
function drawAllCoverage() {
   	show_coverage['apogee'] = true;
   	show_coverage['boss'] = true;
		eraseCoverage();
		covers = new Array();
    for (i = 0; i < robot_obj[iexp].xPos.length; i++) {
				covers.push(drawCoverage(robot_obj[iexp], i));
    }
}

// Remove coverage drawing elements and erase list
function eraseTargets() {
    ntargets = targets.length;
    for (i = 0; i < ntargets; i++) {
				targets[0].remove();
				targets.shift();
    }
}

// Draw a target, return graphics element
function drawTarget(target_obj, i) {
		var [xpix, ypix] = xymm2pix(target_obj.x[i], target_obj.y[i]);
		var target = stage.circle(xpix, ypix, 2);
		if(target_obj.fiberID[i] == 1) {
				target.fill('red', 0.2);
    } else {
				target.fill('blue', 0.2);
	  }
		target.listen('mouseover', function (e) {
        text_targetid.text('Target ID = ' + target_obj.id[i]);
		});
		target.listen('mouseout', function (e) {
        text_targetid.text('Target ID =');
		});
		return target;
}

// Draw all the targets
function drawAllTargets() {
    show_targets = true;
		eraseTargets();
    targets = new Array();
    for (i = 0; i < target_obj.x.length; i++) {
				targets.push(drawTarget(target_obj, i));
    }
}

// Return position of alpha end
function alphaEndXY(xpix, ypix, alpha) {
	  alphaEndX = xpix + alphaLen * Math.cos(alpha / 180. * Math.PI);
	  alphaEndY = ypix - alphaLen * Math.sin(alpha / 180. * Math.PI);
		return [alphaEndX, alphaEndY];
}

// Draw a robot, return array with graphics elements
function drawRobot(robot_obj, target_obj, i) {
		var [xpix, ypix] = xymm2pix(robot_obj.xPos[i], robot_obj.yPos[i]);
		var alpha = 0.;
		var beta = 180.;
		
		// draw center of robot
		var center = stage.circle(xpix, ypix, 1);
		
		// draw the alpha arm
		var alphaArm = stage.rect(0, - alphaWid / 2, alphaLen, alphaWid);
		alphaArm.translate(xpix, ypix);
		alphaArm.rotate(- alpha, xpix, ypix);
		alphaArm.fill('yellow', 0.5);
		
	  var [alphaEndX, alphaEndY] = alphaEndXY(xpix, ypix, alpha);
	  alphaEnd = stage.circle(alphaEndX, alphaEndY, 1);

    var betaArm = stage.rect(0, - betaWid / 2, betaLen, betaWid);
	  betaArm.translate(alphaEndX, alphaEndY);
	  betaArm.rotate(- (beta + alpha), alphaEndX, alphaEndY);
		if(robot_obj.isAssigned[i]) {
        if(target_obj.fiberID == 2) {
						var betaColor = 'blue';
        } else {
						var betaColor = 'red';
		    }
    } else {
			  var betaColor = 'grey';
    }
		if(robot_obj.isCollided[i]) {
				var betaOpacity = 1.;
    } else {
			  var betaOpacity = 0.4;
		}		
	  betaArm.fill(betaColor, betaOpacity);
		
		betaArm.listen('mouseover', function (e) {
        text_robotid.text('Robot ID = ' + robot_obj.id[i]);
		});
		
		betaArm.listen('mouseout', function (e) {
        text_robotid.text('Robot ID = ');
		});
		
		var robot = new Array();
		robot.push(alphaArm);
		robot.push(alphaEnd);
		robot.push(betaArm);
		robot.push(alpha);
		robot.push(beta);
		return robot;
}

// Remove coverage drawing elements and erase list
function eraseRobots() {
    nrobots = robots.length;
    for (i = 0; i < nrobots; i++) {
				robots[0][0].remove();
				robots[0][1].remove();
				robots[0][2].remove();
				for (j = 0; j < robots[0].length; j++) {
						robots[0].shift();
				}
				robots.shift();
    }
}

// Draw all the robots
function drawAllRobots() {
    show_robots['apogee'] = true;
    show_robots['boss'] = true;
		eraseRobots();
		robots = new Array();
    for (i = 0; i < robot_obj[iexp].xPos.length; i++) {
		  	robots.push(drawRobot(robot_obj[iexp], target_obj, i));
    }
}

// Reset the arms for the robots
function setAllRobots() {
    for (i = 0; i < robot_obj[iexp].xPos.length; i++) {
				var [xpix, ypix] = xymm2pix(robot_obj[iexp].xPos[i], robot_obj[iexp].yPos[i]);
		    var alpha = robot_obj[iexp].alpha[i];
		    var beta = robot_obj[iexp].beta[i];
				var alphaEndX, alphaEndY;
		    robots[i][0].rotate(robots[i][3], xpix, ypix);
		    robots[i][0].rotate(- alpha, xpix, ypix);
				[alphaEndX, alphaEndY] = alphaEndXY(xpix, ypix, robots[i][3]);
		    robots[i][1].translate(- alphaEndX, - alphaEndY)
		    robots[i][2].translate(- alphaEndX, - alphaEndY)
	  	  robots[i][2].rotate(robots[i][4] + robots[i][3], alphaEndX, alphaEndY);
				[alphaEndX, alphaEndY] = alphaEndXY(xpix, ypix, alpha);
		    robots[i][1].translate(alphaEndX, alphaEndY)
		    robots[i][2].translate(alphaEndX, alphaEndY)
	  	  robots[i][2].rotate(- (beta + alpha), alphaEndX, alphaEndY);
				robots[i][3] = alpha
				robots[i][4] = beta
				if(robot_obj[iexp].isAssigned[i]) {
						if(target_obj.fiberID[robot_obj[iexp].assignedTargetInd[i]] == 2) {
								var betaColor = 'blue';
						} else {
								var betaColor = 'red';
						}
				} else {
						var betaColor = 'grey';
				}
				if(robot_obj[iexp].isCollided[i]) {
						var betaOpacity = 1.;
				} else {
						var betaOpacity = 0.4;
				}		
	  		robots[i][2].fill(betaColor, betaOpacity);
    }
}

function toggleCoverage(type) {
    if(show_coverage[type]) {
				newWeight = 'normal';
				newVisible = false;					  
		} else {
				newWeight = 'bold';
				newVisible = true;					  
		}
    text_coverage[type].fontWeight(newWeight);
    for (i in covers) {
				if(((type == 'apogee') && (robot_obj[iexp].hasApogee[i] > 0)) || ((type == 'boss') && (robot_obj[iexp].hasApogee[i] == 0))) {
						covers[i].visible(newVisible);
        }
    }
    show_coverage[type] = newVisible;
}

function drawTextCoverage() {
		text_coverage['all']	= control.text(0, 10, 'Coverage');
		text_coverage['all'].fontSize('16px');
		text_coverage['all'].fontWeight('bold');
		text_coverage['boss'] = control.text(10, 40, 'BOSS');
		text_coverage['boss'].fontSize('14px');
		text_coverage['boss'].fontWeight('bold');
		text_coverage['apogee'] = control.text(10, 60, 'APOGEE+BOSS');
		text_coverage['apogee'].fontSize('14px');
		text_coverage['apogee'].fontWeight('bold');
		show_coverage["boss"] = true
		show_coverage["apogee"] = true
		
		text_coverage['all'].listen('click', function(e) {
				toggleCoverage('apogee');
				toggleCoverage('boss');
		});
		text_coverage['apogee'].listen('click', function(e) {
				toggleCoverage('apogee');
		});
		text_coverage['boss'].listen('click', function(e) {
				toggleCoverage('boss');
		});
}
		
function toggleRobots(type) {
		if(show_robots[type]) {
				newWeight = 'normal';
				newVisible = false;					  
		} else {
				newWeight = 'bold';
				newVisible = true;					  
		}
		text_robots[type].fontWeight(newWeight);
		for (i in robots) {
				if(((type == 'apogee') &&
						(robot_obj[iexp].hasApogee[i] > 0)) ||
					 ((type == 'boss') &&
						(robot_obj[iexp].hasApogee[i] == 0))) {
						robots[i][0].visible(newVisible);
						robots[i][1].visible(newVisible);
						robots[i][2].visible(newVisible);
				}
		}
		show_robots[type] = newVisible;
}

function drawTextRobots() {
		text_robots['all']	= control.text(0, 90, 'Robots');
		text_robots['all'].fontSize('16px');
		text_robots['all'].fontWeight('bold');
		text_robots['boss'] = control.text(10, 120, 'BOSS');
		text_robots['boss'].fontSize('14px');
		text_robots['boss'].fontWeight('bold');
		text_robots['apogee'] = control.text(10, 140, 'APOGEE+BOSS');
		text_robots['apogee'].fontSize('14px');
		text_robots['apogee'].fontWeight('bold');
		show_robots["boss"] = true
		show_robots["apogee"] = true

		text_robots['all'].listen('click', function(e) {
				toggleRobots('apogee');
				toggleRobots('boss');
		});
		text_robots['apogee'].listen('click', function(e) {
				toggleRobots('apogee');
		});
		text_robots['boss'].listen('click', function(e) {
				toggleRobots('boss');
		});
}
