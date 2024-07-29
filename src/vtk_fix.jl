
function Gridap.Visualization.create_vtk_file(
  trian::Grid, filebase; celldata=Dict(), nodaldata=Dict())

  points = Gridap.Visualization._vtkpoints(trian)
  cells = Gridap.Visualization._vtkcells(trian)
  vtkfile = Gridap.Visualization.vtk_grid(filebase, points, cells, compress=false, append=false)

  if num_cells(trian)>0
    for (k,v) in celldata
      Gridap.Visualization.vtk_cell_data(vtkfile, Gridap.Visualization._prepare_data(v), k)
    end
    for (k,v) in nodaldata
      Gridap.Visualization.vtk_point_data(vtkfile, Gridap.Visualization._prepare_data(v), k)
    end
  end

  return vtkfile
end

function Gridap.Visualization.create_pvtk_file(
  trian::Grid, filebase;
  part, nparts, ismain=(part==1), celldata=Dict(), nodaldata=Dict())

  points = Gridap.Visualization._vtkpoints(trian)
  cells = Gridap.Visualization._vtkcells(trian)
  vtkfile = Gridap.Visualization.pvtk_grid(filebase, points, cells, compress=false, append=false;
                      part=part, nparts=nparts, ismain=ismain)

  if num_cells(trian) > 0
    for (k, v) in celldata
      vtkfile[k, VTKCellData()] = Gridap.Visualization._prepare_data(v)
    end
    for (k, v) in nodaldata
      vtkfile[k, VTKPointData()] = Gridap.Visualization._prepare_data(v)
    end
  end
  return vtkfile
end
