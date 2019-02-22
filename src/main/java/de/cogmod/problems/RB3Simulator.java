package de.cogmod.problems;

import java.awt.AlphaComposite;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import javax.imageio.IIOImage;
import javax.imageio.ImageIO;
import javax.imageio.ImageWriteParam;
import javax.imageio.ImageWriter;
import javax.imageio.stream.FileImageOutputStream;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.Timer;

import de.jannlab.examples.tools.Vector2d;
import de.jannlab.math.MathTools;
import de.jannlab.math.Matrix;



public class RB3Simulator implements CSCProblemAndOutInMapInterface {
	//
	final Random rnd = new Random(100L);

	public enum RB3Mode{Rocket, Glider, GliderGravity, Stepper, Stepper2Dir, Car};

	private Vector2d gvec;
	private double   g;
	private final double   mass;
	private double   radius;
	private double   leftborder;
	private double   rightborder;
	private double   topborder;
	private double   maxthrust;
	private double orientation;

	private double[] initialPosition;

	private final int sensorSize;
	private final boolean doDelta;
	
	private int numThrusts;
	private double[]   thrusts; 
	private Vector2d[] thrustsDir;
	private Vector2d thrustCurrDir;
	private Vector2d thrustForcevec;

	private final double[][] minMaxMotorValues = new double[][]{{0,0,0,0},{1,1,1,1}};
	
	private Vector2d forcesumvec;
	private Vector2d previousPosition;
	private Vector2d currentPosition;
	private Vector2d velocity;
	private Vector2d acceleration;

	private List<double[]> obstacles = new ArrayList<double[]>();
	private List<double[]> monsters = new ArrayList<double[]>();

	// Friction factor... ergo: 1=no friction
	private double floorFriction = .9;
	private double ceilingFriction = 1;
	private double sideFriction = 1;

	// flag for further printing options
	private boolean print = true;
	//		graphics setup
	private final Color backgroundcolor   = (print)?(Color.WHITE):(Color.BLACK) ;
	private final Color thrustcolor       = (print)?(new Color(255, 179, 0)):(new Color(255, 255, 0));
	private final Color obstaclecolor     = (print)?(new Color(126, 126,126)):(new Color(120, 120, 120));
	private final Color trajectorycolor   = (new Color(80, 150, 180));
	private final Color targetcolor       = (print)?(new Color( 0, 200, 0)):(Color.CYAN);
	private final Color ballestimatecolor = (print)?(new Color(216, 8, 0)):(new Color(255, 50, 50));
	final double drawscale = 250.0;

	// flag to control drawing the visualization.
	private boolean doDraw = true;

	//	maintenance of image components and JPanel	
	private final BufferedImage panelImage;
	private final Graphics2D panelG2D;
	private final JPanel panel;

	private final double deltaTime;
	private long count = 0;
	private final Vector2d currentTarget;

	private RB3Mode currentMode;

	private final BufferedImage pcOpenImage;
	private final BufferedImage pcClosedImage;

	private final BufferedImage rocketShip;
	private final BufferedImage spaceGlider;
	private final BufferedImage stepper2DirImage;
	private final BufferedImage stepper4DirImage;
	private final BufferedImage carImage;
	private final BufferedImage carImageL;

	private Matrix stateseqshort = null;
	
	private int imageIndex = 0;
	public boolean doSaveImage = false;
	
	public void setStateSeqshort(Matrix seq) {
		this.stateseqshort =  seq;
	}
	
	public BufferedImage loadImage(String fileName) {
		BufferedImage img = null;
		try {
			img = ImageIO.read(new File(fileName));
			img = resizeImage(img, (int)(2.*this.radius*drawscale), (int)(2.*this.radius*drawscale));
		} catch (IOException e) {
			System.out.println("Failed to read file"+e+" name: "+fileName);
		}
		return img;
	}

	/**
	 * This function resize the image file and returns the BufferedImage object that can be saved to file system.
	 */
	public static BufferedImage resizeImage(final BufferedImage image, int width, int height) {
		final BufferedImage bufferedImage = new BufferedImage(width, height, image.getType());
		final Graphics2D graphics2D = bufferedImage.createGraphics();
		graphics2D.setBackground(new Color(0,0,0,1));

		graphics2D.setComposite(AlphaComposite.Src);
		graphics2D.setRenderingHint(RenderingHints.KEY_INTERPOLATION,RenderingHints.VALUE_INTERPOLATION_BILINEAR);
		graphics2D.setRenderingHint(RenderingHints.KEY_RENDERING,RenderingHints.VALUE_RENDER_QUALITY);
		graphics2D.setRenderingHint(RenderingHints.KEY_ANTIALIASING,RenderingHints.VALUE_ANTIALIAS_ON);
		graphics2D.drawImage(image, 0, 0, width, height, null);
		graphics2D.dispose();

		return bufferedImage;
	}

	public void setCurrentMode(final RB3Mode currentMode) {
		this.currentMode = currentMode;
		switch(currentMode) {
		case Glider:
			this.g = 0;
			RB3Simulator.computeDownGravityForceVec(this.mass, this.g, this.gvec);
			this.floorFriction = .9;
			this.ceilingFriction = .9;
			this.sideFriction = .9;
			this.thrusts = new double[4];
			this.thrustsDir = new Vector2d[4];
			this.thrustsDir[0] = Vector2d.normalize(new Vector2d(1.0, 1.0));
			this.thrustsDir[1] = Vector2d.normalize(new Vector2d(-1.0, 1.0));
			this.thrustsDir[2] = Vector2d.normalize(new Vector2d(1.0, -1.0));
			this.thrustsDir[3] = Vector2d.normalize(new Vector2d(-1.0, -1.0));
			this.numThrusts = 4;
			break;
		case GliderGravity:
			this.g = 9.81;
			RB3Simulator.computeDownGravityForceVec(this.mass, this.g, this.gvec);
			this.floorFriction = .9;
			this.ceilingFriction = .9;
			this.sideFriction = .9;
			this.thrusts = new double[4];
			this.thrustsDir = new Vector2d[4];
			this.thrustsDir[0] = Vector2d.normalize(new Vector2d(1.0, 1.0));
			this.thrustsDir[1] = Vector2d.normalize(new Vector2d(-1.0, 1.0));
			this.thrustsDir[2] = Vector2d.normalize(new Vector2d(1.0, -1.0));
			this.thrustsDir[3] = Vector2d.normalize(new Vector2d(-1.0, -1.0));
			this.numThrusts = 4;
			break;
		case Rocket:
			this.g = 9.81;
			RB3Simulator.computeDownGravityForceVec(this.mass, this.g, this.gvec);
			this.orientation = 0;
			this.floorFriction = .9;
			this.ceilingFriction = .95;
			this.sideFriction = .95;
			this.numThrusts = 2;
			this.thrusts = new double[numThrusts];
			this.thrustsDir = new Vector2d[numThrusts];
			this.thrustsDir[0] = Vector2d.normalize(new Vector2d(1.0, 1.0));
			this.thrustsDir[1] = Vector2d.normalize(new Vector2d(-1.0, 1.0));
			break;
		case Stepper:
			this.g = 0;
			RB3Simulator.computeDownGravityForceVec(this.mass, this.g, this.gvec);
			this.orientation = 0;
			this.velocity = new Vector2d(0,0);
			this.floorFriction = 1;
			this.ceilingFriction = 1;
			this.sideFriction = 1;
			this.thrusts = new double[4];
			this.thrustsDir = new Vector2d[4];
			this.thrustsDir[0] = Vector2d.normalize(new Vector2d(1.0, 1.0));
			this.thrustsDir[1] = Vector2d.normalize(new Vector2d(-1.0, 1.0));
			this.thrustsDir[2] = Vector2d.normalize(new Vector2d(1.0, -1.0));
			this.thrustsDir[3] = Vector2d.normalize(new Vector2d(-1.0, -1.0));
			this.numThrusts = 4;
			break;
		case Stepper2Dir:
			this.g = 0;
			RB3Simulator.computeDownGravityForceVec(this.mass, this.g, this.gvec);
			this.velocity = new Vector2d(0,0);
			this.floorFriction = 1;
			this.ceilingFriction = 1;
			this.sideFriction = 1;
			this.numThrusts = 2;
			this.thrusts = new double[numThrusts];
			this.thrustsDir = new Vector2d[numThrusts];
			this.thrustsDir[0] = Vector2d.normalize(new Vector2d(1.0, 1.0));
			this.thrustsDir[1] = Vector2d.normalize(new Vector2d(-1.0, 1.0));
			break;
		case Car:
			this.g = 9.81;
			RB3Simulator.computeDownGravityForceVec(this.mass, this.g, this.gvec);
			this.orientation = 0;
			this.floorFriction = .8;
			this.ceilingFriction = 1;
			this.sideFriction = 1;
			this.numThrusts = 2;
			this.thrusts = new double[numThrusts];
			this.thrustsDir = new Vector2d[numThrusts];
			this.thrustsDir[0] = Vector2d.normalize(new Vector2d(1.0, 0));
			this.thrustsDir[1] = Vector2d.normalize(new Vector2d(-1.0, 0));
			break;
		}
	}

	@Override
	public String getProblemName() {
		return "RB3 Simulator ("+this.currentMode+")";
	}

	@Override
	public int getSensorInputSize() {
		return this.sensorSize;
	}


	@Override
	public int getSensorOutputSize() {
		return this.sensorSize;
	}

	@Override
	public int getMotorSize() {
		return 4;
	}

	@Override
	public double[][] getMinMaxMotorValues() {
		return this.minMaxMotorValues;
	}


	@Override
	public void setCurrentTarget(double[] target) {
		this.currentTarget.x = target[0];
		this.currentTarget.y = target[1];
	}

	@Override
	public void executeAndGet(double[] motorCommands, double[] sensorOutput, double[] nextSensorInput) {
		MathTools.clamp(motorCommands, 0, 1);
		this.thrusts[0] = motorCommands[0];
		this.thrusts[1] = motorCommands[1];
		if(this.currentMode == RB3Mode.Glider || this.currentMode == RB3Mode.GliderGravity || this.currentMode == RB3Mode.Stepper) {
			this.thrusts[2] = motorCommands[2];
			this.thrusts[3] = motorCommands[3];
		}else{
			addOrientation( (motorCommands[3] - motorCommands[2]) * Math.PI);
		}
		update();
		if(doDraw) {
			drawPanel(null, null, this.stateseqshort);
		}
		if(this.doDelta) {
			sensorOutput[0] = this.currentPosition.x - this.previousPosition.x;
			sensorOutput[1] = this.currentPosition.y - this.previousPosition.y;
		}else{
			sensorOutput[0] = this.currentPosition.x;
			sensorOutput[1] = this.currentPosition.y;			
		}
		nextSensorInput[0] = this.currentPosition.x;
		nextSensorInput[1] = this.currentPosition.y;
	}

	@Override
	public void mapStateAndOutputToNextInput(double[] sensorInput, double[] sensorOutput, double[] nextSensorInput) {
		if(this.doDelta) {
			for(int i=0; i<this.sensorSize; i++) {
				nextSensorInput[i] = sensorInput[i] + sensorOutput[i];
			}			
		}else{
			for(int i=0; i<this.sensorSize; i++) {
				nextSensorInput[i] = sensorOutput[i];
			}
		}
	}

	
	private void setAction(final int index, final double value) {
		assert(index>=0 && index<4);
		double cValue = MathTools.clamp(value, 0.0, 1.0);
		if(this.currentMode == RB3Mode.Glider || this.currentMode == RB3Mode.GliderGravity || this.currentMode == RB3Mode.Stepper) {
			switch(index) {
			case 0: this.thrusts[0] = cValue; break;
			case 1: this.thrusts[3] = cValue; break;
			case 2: this.thrusts[2] = cValue; break;
			case 3: this.thrusts[1] = cValue; break;
			}
		}else{
			if(index >= 2) {
				if(index == 2) {
					addOrientation(-.2* cValue * Math.PI);
				}else{
					addOrientation(.2 * cValue * Math.PI);
				}
			}else{
				this.thrusts[index] = cValue;
			}
		}
	}

	public enum Mode {
		FREE(""),
		OBSTACLE(".obstacle"),
		OBSTACLE2(".obstacle2"),
		OBSTACLE3(".obstacle3"),
		OBSTACLE4(".obstacle4");
		private String tag = "";
		private Mode(final String tag) {
			this.tag = tag;
		}
		public String getTag() { return tag;}
	}

	private static void computeDownGravityForceVec(final double mass, final double g, final Vector2d forcevec) {
		forcevec.x = 0.0f;
		forcevec.y = -1.0f;
		// F = m * a
		Vector2d.mul(forcevec, (double)(mass * g), forcevec);
	}

	@Override
	public void reset() {
		this.currentPosition.x = this.previousPosition.x = this.initialPosition[0];
		this.currentPosition.y = this.previousPosition.y = this.initialPosition[1] + 0.2 + this.radius;
		this.forcesumvec.x = 0; this.forcesumvec.y = 0;
		this.velocity.x = 0; this.velocity.y = 0;
		this.acceleration.x = 0; this.acceleration.y = 0;
	}

	public boolean isDoDraw() {
		return doDraw;
	}

	@Override
	public void setDoDraw(boolean doDraw) {
		this.doDraw = doDraw;
	}

	private double[] join(final double ...x) {
		return x;
	}

	public void clearObstacles() {
		this.obstacles.clear();
	}

	public void addObstacle(final double x1, final double y1, final double x2, final double y2) {
		this.obstacles.add(join(Math.min(x1, x2), Math.min(y1, y2), Math.max(x1,  x2), Math.max(y1, y2)));
	}

	public void clearMonsters() {
		this.monsters.clear();
	}

	public void addMonster(final double x, final double y) {
		this.monsters.add(new double[]{x, y, this.radius});
	}

	public double getTopBorder() {
		return this.topborder;
	}

	public double getLeftBorder() {
		return this.leftborder;
	}

	public double getRightBorder() {
		return this.rightborder;
	}

	public double getRadius() {
		return this.radius;
	}

	public Vector2d getPosition() {
		return this.currentPosition;
	}

	public static void getThrustDirection(Vector2d thrustDir, double angle, Vector2d thrustCurrDir) {
		Vector2d.rotate(thrustDir, angle, thrustCurrDir);
	}

	/**
	 * Simulates one update step
	 * 
	 */
	private void update() {
		// update thrust force vec.
		// thrust to thrust direction vector
		this.forcesumvec.x = gvec.x;
		this.forcesumvec.y = gvec.y;
		for(int i=0; i<this.numThrusts; i++) {
			this.thrusts[i] = MathTools.clamp(this.thrusts[i], this.minMaxMotorValues[0][i], this.minMaxMotorValues[1][i]);
			Vector2d.rotate(this.thrustsDir[i], this.orientation, this.thrustCurrDir);
			Vector2d.mul(this.thrustCurrDir,  (double)(this.thrusts[i] * this.maxthrust), this.thrustForcevec);
			Vector2d.add(this.forcesumvec, this.thrustForcevec, this.forcesumvec);
		}
		// compute acceleration vector (a = F / m)
		Vector2d.mul(this.forcesumvec, (double)(1.0 / this.mass), this.acceleration);
		//
		// would the ball with the acceleration and hypothetical velocity
		// exceed the ground line?
		// -> consequent velocity without borders.
		final Vector2d hypvel = new Vector2d(
				(double)(this.velocity.x + this.acceleration.x * this.deltaTime),
				(double)(this.velocity.y + this.acceleration.y * this.deltaTime)
				);
		// -> consequent position without borders.
		final Vector2d hyppos = new Vector2d(
				(double)(this.currentPosition.x + hypvel.x * this.deltaTime),
				(double)(this.currentPosition.y + hypvel.y * this.deltaTime)    
				);
		// new velocities and position without border
		double nvelx = hypvel.x; 
		double nvely = hypvel.y;
		double nposx = hyppos.x;
		double nposy = hyppos.y;
		// differences between bottom and top border
		final double ylodiff = hyppos.y - this.radius; // bottom border = 0;
		final double yhidiff = hyppos.y - (this.topborder - this.radius);
		// check borders
		if (ylodiff <= 0) { // down at floor border
			nvely = 0.0f; 
			nvelx = (double)(hypvel.x * floorFriction);
			nposy = (double)(this.radius);
		} else if (yhidiff >= 0) { // up at top ceiling border
			nvely = 0.0f;
			nvelx = (double)(hypvel.x * ceilingFriction);
			nposy = (double)(this.topborder - this.radius);
		}
		// differences between left and right border
		final double xlediff = hyppos.x - (this.leftborder + this.radius);
		final double xridiff = hyppos.x - (this.rightborder - this.radius);
		// check borders
		if (xlediff <= 0) { // too far left
			nvelx = 0.0f;
			nvely = (double)(hypvel.y * sideFriction);
			nposx = (double)(this.leftborder + this.radius);
		} else if (xridiff >= 0) {
			nvelx = 0.0f;
			nvely = (double)(hypvel.y * sideFriction);
			nposx = (double)(this.rightborder - this.radius);
		}        
		// Obstacle behavior.
		for (double[] obstacle : this.obstacles) {
			final double x1 = obstacle[0]; 
			final double y1 = obstacle[1]; 
			final double x2 = obstacle[2]; 
			final double y2 = obstacle[3];
			// in obstacle?
			if (    (nposx >= (x1 - this.radius)) &&
					(nposx <= (x2 + this.radius)) &&
					(nposy >= (y1 - this.radius)) &&
					(nposy <= (y2 + this.radius))
					) {
				// where is the collision?
				final double dy1 = Math.abs((nposy + this.radius) - y1);
				final double dy2 = Math.abs((nposy - this.radius) - y2);
				final double dx1 = Math.abs((nposx + this.radius) - x1);
				final double dx2 = Math.abs((nposx - this.radius) - x2);
				// smallest border violation is the one that is corrected (violated)
				final double min = MathTools.min(dx1, dx2, dy1, dy2);
				if (dy2 <= min) { // top border violation
					nvely = 0.0f;
					nvelx = (double)(nvelx * floorFriction);
					nposy = (double)(y2 + this.radius);
				} else if (dy1 <= min) { // bottom border violation
					nvely = 0.0f;
					nvelx = (double)(nvelx * ceilingFriction);
					nposy = (double)(y1 - this.radius);
				} else if (dx1 <= min) { // left-side violation
					nvelx = 0.0f;
					nvely = (double)(nvely * sideFriction);
					//
					nposx = (double)(x1 - this.radius);
				} else if (dx2 <= min) { // right-side violoation
					nvelx = 0.0f;
					nvely = (double)(nvely * sideFriction);
					nposx = (double)(x2 + this.radius);
				}
			}
		} // done with obstacle handling.
		// Monster behavior.
		for (double[] monster : this.monsters) {
			double dis = Math.sqrt((nposx-monster[0])*(nposx-monster[0])+
					(nposy-monster[1])*(nposy-monster[1]));
			if(dis < monster[2] + this.radius) {
				reset();
				return;
			}
//			System.out.println(""+dis+ " vs." + (monster[2] + this.radius));
		}

		//
		if(this.currentMode==RB3Mode.Stepper2Dir || this.currentMode==RB3Mode.Stepper) {
			this.velocity.x = 0;
			this.velocity.y = 0;
		}else{
			this.velocity.x = nvelx;
			this.velocity.y = nvely;			
		}
		this.previousPosition.x = this.currentPosition.x;
		this.previousPosition.y = this.currentPosition.y;
		this.currentPosition.x = nposx;
		this.currentPosition.y = nposy;
	} // done with update step

	public List<double[]> getObstacles() {
		return this.obstacles;
	}

	public RB3Simulator(
			final RB3Mode startMode,
			final boolean doDelta,
			final double mass,
			final double radius,
			final double maxthrust,
			final double leftborder,
			final double rightborder,
			final double topborder, 
			final double deltaTime,
			final double[] initPos
			) {
		this.initialPosition = initPos.clone();
		this.sensorSize = 2;
		this.doDelta = doDelta;
		
		this.mass            = mass;
		this.gvec            = new Vector2d();
		this.thrustCurrDir = new Vector2d();
		this.thrustForcevec = new Vector2d();
		this.forcesumvec     = new Vector2d(0.0, 0.0);
		setCurrentMode(startMode);
		this.radius          = radius;
		this.leftborder      = leftborder;
		this.rightborder     = rightborder;
		this.topborder       = topborder;
		this.maxthrust       = maxthrust;
		this.previousPosition = new Vector2d(this.initialPosition[0], this.initialPosition[1] + 0.2 + radius);
		this.currentPosition = new Vector2d(this.initialPosition[0], this.initialPosition[1] + 0.2 + radius);
		this.velocity        = new Vector2d(0.0, 0.0);
		this.acceleration    = new Vector2d(0.0, 0.0);
		this.deltaTime 		 = deltaTime;
		this.currentTarget 	 = new Vector2d(0.0, 0.0);

		this.pcClosedImage = null;
		this.pcOpenImage = null;
		this.rocketShip = loadImage("images/RocketShip.png");
		this.spaceGlider = loadImage("images/SpaceGlider.png");
		this.stepper2DirImage = null;
		this.stepper4DirImage = loadImage("images/Stepper4Dir.png");
		this.carImage = null;
		this.carImageL = null;

		panelImage = new BufferedImage(800, 600, BufferedImage.TYPE_INT_RGB);
		panel = new JPanel() {
			private static final long serialVersionUID = -4307908552010057652L;
			@Override
			protected void paintComponent(final Graphics gfx) {
				super.paintComponent(gfx);
				gfx.drawImage(
						panelImage,  0,  0, 
						panelImage.getWidth(),  panelImage.getHeight(),  null
						);
			}
		};
		this.panelG2D = (Graphics2D)panelImage.getGraphics();
		this.panelG2D.setRenderingHint(
				RenderingHints.KEY_ANTIALIASING,
				RenderingHints.VALUE_ANTIALIAS_ON
				);
		panel.setPreferredSize(new Dimension(panelImage.getWidth(), panelImage.getHeight()));
	}

	public void drawPanel(final Matrix systemstateestimate,
			final Matrix stateseq, final Matrix stateseqshort) {
		count++;
		// 
		final int centerx = panelImage.getWidth() / 2;
		final int groundy = panelImage.getHeight() - 80;
		//
		panelG2D.setColor(this.obstaclecolor);
		panelG2D.fillRect(0, 0, panelImage.getWidth(), panelImage.getHeight());
		//
		panelG2D.setColor(this.backgroundcolor);
		panelG2D.fillRect((int)(centerx + this.drawscale*this.leftborder), 
				(int)(groundy - this.drawscale * this.topborder),
				(int)(this.drawscale * (this.rightborder - this.leftborder)), 
				(int)(this.drawscale * this.topborder));
		//
		panelG2D.drawString(this.currentMode.toString(), 20, panelImage.getHeight()-20);
		//
		// draw obstacles.
		panelG2D.setColor(this.obstaclecolor);
		for (double[] obstacle : this.obstacles) {
			//
			final double x1 = Math.min(obstacle[0], obstacle[2]);
			final double x2 = Math.max(obstacle[0], obstacle[2]);
			final double y1 = Math.min(obstacle[1], obstacle[3]);
			final double y2 = Math.max(obstacle[1], obstacle[3]);
			//
			final int w  = (int)(drawscale * (x2 - x1));
			final int h  = (int)(drawscale * (y2 - y1));
			//
			final int x = centerx + (int)(x1 * drawscale);
			final int y = groundy - (int)(y1 * drawscale) - h;
			panelG2D.fillRect(x, y, w, h);
		}
		//
		// draw monsters.		
		final int monsterrad   = (int)(radius * drawscale);
		for (double[] monster : this.monsters) {
			final int x = centerx + (int)(monster[0] * drawscale);
			final int y = groundy - (int)(monster[1] * drawscale);
			if((long)(count/10)%2 == 0) {
				panelG2D.drawImage(this.pcOpenImage, null, (int)(x-monsterrad), (int)(y-monsterrad));
			}else{
				panelG2D.drawImage(this.pcClosedImage, null, (int)(x-monsterrad), (int)(y-monsterrad));
			}
		}
		//
		final Vector2d ballpos = getPosition();
		//
		final int ballposx  = centerx + (int)(ballpos.x * drawscale);
		final int ballposy  = groundy - (int)(ballpos.y * drawscale);
		final int ballrad   = (int)(radius * drawscale);
		//
		final int targetposx  = centerx + (int)(currentTarget.x * drawscale);
		final int targetposy  = groundy - (int)(currentTarget.y * drawscale);
		//
		panelG2D.setColor(this.thrustcolor);
		//
		int[] polyx = {0,0,0};
		int[] polyy = {0,0,0};
		//
		double r = 0;

		for(int i=0; i<this.numThrusts; i++) {
			r = 0.1 * rnd.nextGaussian(); // making the thrust visualization lively
			RB3Simulator.getThrustDirection(this.thrustsDir[i], orientation, this.thrustCurrDir);
			polyx[0] = ballposx - (int)((1.0 + r) * this.thrustCurrDir.x * this.thrusts[i] * drawscale * 4.0 * getRadius());
			polyy[0] = ballposy + (int)((1.0 + r) * this.thrustCurrDir.y * this.thrusts[i] * drawscale * 4.0 * getRadius());
			polyx[1] = ballposx;
			polyy[1] = ballposy - 5;
			polyx[2] = ballposx;
			polyy[2] = ballposy + 5;
			//
			panelG2D.fillPolygon(polyx, polyy, 3);
		}
		//
		// draw controlled object...
		switch(this.currentMode) {
		case Car:
			if(this.velocity.x >= 0) {
				panelG2D.drawImage(this.carImage, null, ballposx-ballrad, ballposy-ballrad);
			}else{
				panelG2D.drawImage(this.carImageL, null, ballposx-ballrad, ballposy-ballrad);
			}
			break;
		case Glider: case GliderGravity:
			panelG2D.drawImage(this.spaceGlider, null, ballposx-ballrad, ballposy-ballrad);
			break;
		case Rocket:
			panelG2D.drawImage(this.rocketShip, null, ballposx-ballrad, ballposy-ballrad);
			break;
		case Stepper2Dir:
			AffineTransform tx = AffineTransform.getRotateInstance(-this.orientation, this.pcOpenImage.getWidth()/2, this.pcOpenImage.getHeight()/2);
			AffineTransformOp op = new AffineTransformOp(tx, AffineTransformOp.TYPE_BILINEAR);
			panelG2D.drawImage(op.filter(this.stepper2DirImage, null), null, ballposx-ballrad, ballposy-ballrad);
			break;
		case Stepper:
			panelG2D.drawImage(this.stepper4DirImage, null, ballposx-ballrad, ballposy-ballrad);
			break;
		}
		//
		if(systemstateestimate!=null) {
			//
			// draw ball estimate.
			//
			panelG2D.setStroke(new BasicStroke(3));
			panelG2D.setColor(this.ballestimatecolor);
			final int ballposestimatex = centerx + (int)(
					systemstateestimate.data[0] * 
					drawscale
					);
			final int ballposestimatey = groundy - (int)(
					systemstateestimate.data[1] * 
					drawscale
					);
			panelG2D.drawOval(ballposestimatex - ballrad, ballposestimatey - ballrad, ballrad * 2, ballrad * 2);
		}
		//    
		panelG2D.setColor(this.targetcolor);
		panelG2D.setStroke(new BasicStroke(2));
		//
		final int targetradius  = 13;
		final int targetradiusi = 9;
		panelG2D.drawOval(targetposx - targetradiusi, targetposy - targetradiusi, 2 * targetradiusi,  2 * targetradiusi);
		panelG2D.drawLine(targetposx, targetposy, targetposx, targetposy);
		panelG2D.drawLine(targetposx, targetposy - (targetradius - 2), targetposx, targetposy - 4);
		panelG2D.drawLine(targetposx, targetposy + (targetradius - 2), targetposx, targetposy + 4);
		panelG2D.drawLine(targetposx - (targetradius - 2), targetposy, targetposx - 4, targetposy);
		panelG2D.drawLine(targetposx + (targetradius - 2), targetposy, targetposx + 4, targetposy);
		//g.drawOval(targetposx - 3, targetposy - 3, 7, 7);
		//g.fillOval(targetposx - 3, targetposy - 3, 7, 7);

		if(stateseq!=null) {
			//
			// draw hypothetical long trajectory.
			//
			int lastx = ballposx;
			int lasty = ballposy;
			panelG2D.setColor(this.trajectorycolor);
			for (int t = 0; t < stateseq.rows; t++) {
				final int x = centerx + (int)(stateseq.get(t, 0) * drawscale);
				final int y = groundy - (int)(stateseq.get(t, 1) * drawscale);
				panelG2D.drawLine(lastx, lasty, x, y);
				//
				lastx = x;
				lasty = y;
			}
		}
		if(stateseqshort!=null) {
			//
			// draw hyptothetical short trajectory.
			//
			int lastx = ballposx;
			int lasty = ballposy;
			panelG2D.setColor(Color.RED);
			lastx = ballposx;
			lasty = ballposy;
			for (int t = 0; t < stateseqshort.rows; t++) {
				final int x = centerx + (int)(stateseqshort.get(t, 0) * drawscale);
				final int y = groundy - (int)(stateseqshort.get(t, 1) * drawscale);
				panelG2D.drawLine(lastx, lasty, x, y);
				//
				lastx = x;
				lasty = y;
			}
		}
		panel.repaint();
		
		if(doSaveImage) {
			panelG2D.drawImage(panelImage, 0, 0, panelImage.getWidth(), panelImage.getHeight(), null);
	        try {
				ImageIO.write(panelImage, "JPG", new File("Image"+this.imageIndex+".jpg"));
				imageIndex++;
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

	@Override
	public JPanel getPanel() {
		return panel;
	}

	private void addOrientation(double changeInOrientation) {
		if(this.currentMode == RB3Mode.Stepper2Dir) {
			this.orientation += changeInOrientation;
			if(orientation < 0) {
				orientation = Math.PI * 2. + orientation;
			}else if(orientation >= 2.*Math.PI) {
				orientation -= 2 * Math.PI;
			}
		}
	}

	private void setNextMode() {
		switch(this.currentMode) {
		case Glider:
			this.setCurrentMode(RB3Mode.GliderGravity);
			break;
		case GliderGravity:
			this.setCurrentMode(RB3Mode.Rocket);
			break;
		case Rocket:
			this.setCurrentMode(RB3Mode.Stepper2Dir);
			break;
		case Stepper2Dir:
			this.setCurrentMode(RB3Mode.Stepper);
			break;
		case Stepper:
			this.setCurrentMode(RB3Mode.Car);
			break;
		case Car:
			this.setCurrentMode(RB3Mode.Glider);
			break;
		}
	}

	@Override
	public void activateKeyListener(JFrame frame) {
		//
		final boolean[] keyup    = {false};
		final boolean[] keydown    = {false};
		final boolean[] keyleft  = {false};
		final boolean[] keyright = {false};
		final boolean[] ai       = {true}; 
		final boolean[] pause    = {false};
		final int[]     ctr      = {1};
		//
		frame.addKeyListener(new KeyAdapter() {
			@Override
			public void keyPressed(KeyEvent e) {
				switch (e.getKeyCode()) {
				case KeyEvent.VK_UP:
					setAction(2, 1);
					keyup[0] = true;
					break;
				case KeyEvent.VK_DOWN:
					setAction(3, 1);
					keydown[0] = true;
					break;
				case KeyEvent.VK_LEFT:
					keyleft[0] = true;
					setAction(0, 1);
					break;
				case KeyEvent.VK_P:
					pause[0] = !pause[0];
					break;
				case KeyEvent.VK_RIGHT:
					keyright[0] = true;
					setAction(1, 1);
					break;
				case KeyEvent.VK_X:
					ai[0] = !ai[0];
					break;
				case KeyEvent.VK_M:
					setNextMode();
					break;
				case KeyEvent.VK_S:
					System.out.println("screenshot");

					try {
						final Iterator<ImageWriter> iter = ImageIO.getImageWritersByFormatName("jpg");
						final ImageWriter writer = (ImageWriter)iter.next();
						// 
						final ImageWriteParam iwp = writer.getDefaultWriteParam();
						iwp.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
						iwp.setCompressionQuality(1);   // an integer between 0 and 1
						//
						IIOImage image = new IIOImage(panelImage, null, null);
						writer.setOutput(new FileImageOutputStream(new File("tmp/screenshots/screenshot"+ String.valueOf(ctr[0]++) + ".jpg")));
						writer.write(null, image, iwp);
						writer.dispose();   
						//
					} catch (IOException e1) {
						e1.printStackTrace();
					}

				}
			}
			@Override
			public void keyReleased(KeyEvent e) {
				System.out.println("Key release:"+e);
				switch (e.getKeyCode()) {
				case KeyEvent.VK_UP:
					setAction(2, 0);
					keyup[0] = false;
					break;
				case KeyEvent.VK_DOWN:
					setAction(3, 0);
					keydown[0] = false;
					break;
				case KeyEvent.VK_LEFT:
					setAction(0, 0);
					keyleft[0] = false;
					break;
				case KeyEvent.VK_RIGHT:
					setAction(1, 0);
					keyright[0] = false;
					break;
				}
			}
		});
	}

	/**
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		//
		final double fps     = 30.0;
		final double dtmsec  = 1000.0 / fps;
		final double dtsec   = dtmsec / 1000.0;
		//
		RB3Simulator rbs = new RB3Simulator(RB3Mode.Rocket, true, 0.1, 0.06, 1.2, -1.5, 1.5, 2.0, dtsec, new double[]{-1.4,0});

		final String caption   = rbs.getProblemName();
		final JFrame frame     = new JFrame(caption);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

		frame.add(rbs.getPanel());
		frame.setResizable(false);
		frame.pack();
		rbs.activateKeyListener(frame);
		frame.setVisible(true);

		Mode mode = Mode.OBSTACLE4;
		//
		rbs.getPosition().y = 0.5f;
		if (mode == Mode.FREE) {
			rbs.setCurrentTarget(new double[]{0, 1});
		} else if (mode == Mode.OBSTACLE) {
			rbs.addObstacle(-0.2, 0.0, 0.2, 1.5);
			rbs.setCurrentTarget(new double[]{.8, .5});
			rbs.getPosition().x = (double)(-0.3 );
			rbs.getPosition().y = (double)(0.1 );
		} else if (mode == Mode.OBSTACLE2){
			rbs.addObstacle(-0.2, 0.0, 0.2, 1.0);
			rbs.addObstacle(-0.5, 1.0, 0.5, 1.5);
			rbs.setCurrentTarget(new double[]{.8, .5});
		} else if (mode == Mode.OBSTACLE3) {
			rbs.addObstacle(-0.7, 0.0, -0.3, 1.5);
			rbs.addObstacle(0.3, 2.0, 0.7, 0.5);
			rbs.setCurrentTarget(new double[]{.9, 1.2});
		} else if (mode == Mode.OBSTACLE4) {
			rbs.addObstacle(-1.3, 2.0, -1.0, 1.5);
			rbs.addObstacle(-1.3, 0.0, -1.0, 1.0);
			rbs.addObstacle(-0.2, 2.0, 0.15, 0.9);
			rbs.addObstacle(-0.2, 0.75, 0.0, 0.5);
			rbs.addObstacle(0.15, 2.0, 0.2, 0.8);
//			rbs.addObstacle(0.0, 0.6, 0.05, 0.5);
			rbs.addObstacle(-0.8, 0.0, -0.5, 1.5);
			rbs.addObstacle(0.5, 0.0, 0.8, 1.2);
			rbs.addObstacle(0.5, 2.0, 0.8, 1.5);
			rbs.addObstacle(1.0, 2.0, 1.3, 0.5);
			rbs.addObstacle(1.0, 0.3, 1.3, 0.0);
			rbs.addMonster(-.4, .1);
			rbs.addMonster(-.2, .1);
			rbs.addMonster(0, .1);
			rbs.addMonster(.2, .1);
			rbs.addMonster(.4, .1);
			rbs.setCurrentTarget(new double[]{1.4, 1.2});
		}
		//
		ActionListener al = new ActionListener() {
			@Override
			public void actionPerformed(final ActionEvent e) {
				rbs.update();
				rbs.drawPanel(null, null, null);
			}
		};

		final Timer timer = new Timer((int)(dtmsec), al);
		timer.start();
	}

}